from dataclasses import dataclass, field
from typing import List, Literal, Optional
import hashlib
import json
from datetime import datetime

@dataclass
class ModelConfig:
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    gpu_memory_utilization: float = 0.6
    prm_paths: List[str] = field(default_factory=lambda: ["zarahall/bias-prm-v3"])
    prm_weights: Optional[List[float]] = None
    prm_combination_strategy: Literal["average_scores", "separate_predictions"] = "average_scores"
    accuracy_weight: float = 0.5
    custom_chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

@dataclass
class DatasetConfig:
    name: str = "heegyu/bbq"
    config: str = "SES"
    split: str = "train"
    start: Optional[int] = None
    end: Optional[int] = None
    num_samples: Optional[int] = None
    
@dataclass
class OutputConfig:
    output_dir: Optional[str] = None
    num_proc: Optional[int] = None
    push_to_hub: bool = True
    hub_dataset_private: bool = True
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

@dataclass
class SearchConfig:
    approach: Literal["best_of_n", "beam_search", "dvts"] = "best_of_n"
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 2
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 2048
    agg_strategy: str = "log_sum"
    beam_width: int = 4
    num_iterations: int = 40
    lookahead: int = 1
    filter_duplicates: bool = False
    sort_completed: bool = False
    data_name: str = "bbq"
    math_temperature: float = 0.5  # Temperature for weighted sum functions in math.py

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    
    def __post_init__(self):
        # Validate configurations
        self._validate_config()
        
        # Generate unique hub dataset ID
        if self.output.push_to_hub:
            self.hub_dataset_id = self._generate_hub_dataset_id()
            
    def _validate_config(self):
        if self.search.approach == "dvts":
            if self.search.n % self.search.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
        
        if self.search.approach == "beam_search" and self.search.search_batch_size != 1:
            raise ValueError("search_batch_size should be 1 for beam_search")
            
        if self.model.prm_weights:
            if len(self.model.prm_weights) != len(self.model.prm_paths):
                raise ValueError("Number of weights must match number of PRM paths")
                
    def _generate_hub_dataset_id(self) -> str:
        """Generate a unique dataset ID based on configuration parameters."""
        # Create a dictionary of relevant parameters
        params = {
            "model": self.model.model_path.split("/")[-1],
            "dataset": self.dataset.name.replace("/", "_"),
            "approach": self.search.approach,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "prms": [p.split("/")[-1] for p in self.model.prm_paths]
        }
        
        # Create a short hash of the parameters
        param_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Construct the dataset ID
        return f"{params['dataset']}__{params['model']}__{params['approach']}__{hash_str}"

    def get_system_prompt(self) -> str:
        """Load system prompt based on dataset name."""
        from sal.config.prompts import get_system_prompt
        return get_system_prompt(self.dataset.name) 