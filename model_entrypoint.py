# EXAMPLE FILE, can be used either directly or via `fastapi dev src/troapis/app.py`
import os

import torch
from transformers import FuyuForCausalLM, FuyuProcessor

device = os.environ.get("DEVICE", "cuda:0")

model_name = "adept/fuyu-8b"
# model_kwargs = {}
model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

# can also load from config if necessary for custom/quicker testing
# model_config = FuyuConfig.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_config(model_config, **model_kwargs).to(device)
model = FuyuForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
processor = FuyuProcessor.from_pretrained(model_name)

processor.tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"


model_info = ModelInfo = {
    "model_name": model_name,
    "model": model,
    "processor": processor,
}

if __name__ == "__main__":
    print("in main of entrypoint")
    import troapis.app

    troapis.app.run_app(model_info_from=model_info)
