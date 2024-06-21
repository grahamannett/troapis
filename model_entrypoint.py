# EXAMPLE FILE
import os

import torch
from transformers import FuyuForCausalLM, FuyuProcessor

model_kwargs = {"torch_dtype": torch.float16}

device = os.environ.get("DEVICE", "cuda:0")

model_name = "adept/fuyu-8b"
# not sure if there is time difference from using "auto" with device_map vs .to("cuda")
model = FuyuForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
processor = FuyuProcessor.from_pretrained(model_name)
gen_kwargs = {
    "stop_strings": ["\n", "<0x0A>"],
    "tokenizer": processor.tokenizer,
}

# smaller/quicker for gpt2,
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# model_name = "gpt2"
# config = AutoConfig.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_config(config).to(device)
# processor = AutoTokenizer.from_pretrained(model_name)
# gen_kwargs = {}


model_info = ModelInfo = {
    "model_name": model_name,
    "model": model,
    "processor": processor,
    "gen_kwargs": gen_kwargs,
}


if __name__ == "__main__":
    print("in main of entrypoint")
    import troapis.app

    troapis.app.run_app(model_info_from=model_info)
