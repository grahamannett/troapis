import os

import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, AutoTokenizer

device = os.environ.get("DEVICE", "cuda:0")
model_name = "google/paligemma-3b-ft-docvqa-896"

# the default model was exported with bfloat16, but the bfloat16 model will nan/inf on generate
model_kwargs = {"torch_dtype": torch.float16, "revision": "float16", "device_map": "auto"}

model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, **model_kwargs).to(device)
processor = PaliGemmaProcessor.from_pretrained(model_name)

# patch the paligemma that does not have chat template
patch_from = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(patch_from)
processor.tokenizer.chat_template = tokenizer.chat_template
processor.tokenizer.apply_chat_template = tokenizer.apply_chat_template

model_info = ModelInfo = {
    "model_name": model_name,
    "model": model,
    # "tokenizer": tokenizer,
    "processor": processor,
    "encode": processor.tokenizer.__call__,  # not using images
}
