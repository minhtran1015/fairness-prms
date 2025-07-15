#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import accumulate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.sal.config import Config

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            "zarahall/llama3-orm-model"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "zarahall/llama3-orm-model",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=8,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)
    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/orm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            answer_scores = []
            for ans in answers:
                conversation = []
                conversation.append({"content": question + " " + ans, "role": "user"})
                conversation.append({"content": "+", "role": "assistant"})
                
                input_ids = self.tokenizer.apply_chat_template(
                    conversation, return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    logits = self.model(input_ids).logits[:, -3, self.candidate_tokens]
                    scores = logits.softmax(dim=-1)[:, 0]
                    answer_scores.append(
                        scores[0].detach().to("cpu", dtype=torch.float32).item()
                    )

            all_scores.append(answer_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        conversations = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation.append({"content": question + " " + ans, "role": "user"})
                conversation.append({"content": "+", "role": "assistant"})
                conversations.append(conversation)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, -3, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:, 0]
                output_scores.extend(
                    scores.detach().to("cpu", dtype=torch.float32).tolist()
                )

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


def load_prm(config: Config) -> PRM:
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)
    
    if config.prm_path == "zarahall/llama3-orm-model":
        return RLHFFlow(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")
