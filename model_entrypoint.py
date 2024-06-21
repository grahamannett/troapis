# EXAMPLE FILE
import os

import torch
from transformers import FuyuForCausalLM, FuyuProcessor

device = os.environ.get("DEVICE", "cuda:0")

model_name = "adept/fuyu-8b"
model_kwargs = {"torch_dtype": torch.float16}

model = FuyuForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
processor = FuyuProcessor.from_pretrained(model_name)
gen_kwargs = {
    # "stop_strings": ["\n", "<0x0A>"],
    # "tokenizer": processor.tokenizer,
}


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
