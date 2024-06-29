from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from troapis.constants import device

# device = "auto"  # device

model_name = "google/gemma-2-9b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)

model_info = ModelInfo = {
    "model_name": model_name,
    "model": model,
    "tokenizer": tokenizer,
}

if __name__ == "__main__":
    print("in main of entrypoint")
    import troapis.app

    troapis.app.run_app(model_info_from=model_info)
