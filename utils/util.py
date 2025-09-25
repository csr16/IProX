import numpy as np
import random
import torch
from transformers import set_seed

llama3_2_chat_template = """
{%- set system_prompt = "You are a helpful AI assistant." -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- system_prompt + '\n\n' -}}
    {%- elif message['role'] == 'user' -%}
        <|start_header_id|>user<|end_header_id|>
        {{- message['content'] | trim + '\n\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        <|start_header_id|>assistant<|end_header_id|>
        {%- if add_generation_prompt -%}
            {% generation %}{{ message['content'] | trim + '\n' }}{% endgeneration %}
        {%- else -%}
            {{- message['content'] | trim + '\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    <|start_header_id|>assistant<|end_header_id|>
{%- endif -%}
"""

qwen2_5_chat_template ="""
{% for message in messages %}
{% if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>
{% endif %}
{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>assistant
{% endif -%}"""

gemma3_chat_template = """
{% for message in messages %}
{% if message['role'] == 'system' -%}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'user' -%}
<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] in ['assistant', 'model'] -%}
<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif %}
{% endfor -%}
{% if add_generation_prompt -%}
<start_of_turn>model
{% endif -%}"""

def setseed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    set_seed(seed)