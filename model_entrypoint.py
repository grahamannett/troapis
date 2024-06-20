# EXAMPLE FILE
import torch
from transformers import FuyuForCausalLM, FuyuProcessor
import os

model_name = "adept/fuyu-8b"
model_kwargs = {"torch_dtype": torch.float16}

device = os.environ.get("DEVICE", "cuda:0")  # or "auto"?
# not sure if there is time difference from using "auto" with device_map vs .to("cuda")
model = FuyuForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

processor = FuyuProcessor.from_pretrained(model_name)

generate_kwargs = {
    "stop_strings": ["\n", "<0x0A>"],
    "tokenizer": processor.tokenizer,
}

generate_kwargs = None
ModelInfo = {
    "model_name": model_name,
    "model": model,
    "processor": processor,
    "generate_kwargs": generate_kwargs,
}


if __name__ == "__main__":
    print("in main of entrypoint")
    from troapis.app import make_app

    app = make_app(
        model_info=ModelInfo,
    )
