# troapis = transformers models + openai api

I am not sure why there isnt something like this already, basically a bare bones way to serve dev model using openai api endpoints/schema. Not intended to be performant or scalable but specifically for getting results from benchmarks like AgentBench with minimal effort but easily adaptable per model as the models I am interested in generally do not have a chat template or encoding without images can cause issues.

# Usage

few ways to use (as I am not clear how I want to be using it yet):

one is via just writing a `model_entrypoint.py` file that loads the model in the file, the server will then check for that file and try and load it, this allows you to easily use `fastapi dev src/troapis/app.py` to run the server and test everything in an interactive way.

Another way can also run the app and pass it in (as seen in model_entrypoint `if __name__ == "__main__":` part)

```python
from troapis.app import run_app
model_info = load_model(...) # would load the model and processor/tokenizer and setup anything else (e.g. chat template or encoding)
run_app(model_info_from=model_info)
```

to use dev can just do `pdm run dev`

The `model_info` object/dict needs the following to work:
### required

- `model_name` - the model name
- `model` - the model object
- `processor` or `tokenizer` - the processor/tokenizer object, if not provided will use the default one from the `model_name`

### optional fields:
- `decode` - the decoding function, if not provided will try to use `processor.decode` or `tokenizer.decode`
- `decode_kwargs` - the decoding function kwargs, if not provided will be `"skip_special_tokens": True`
- `encode` - the encoding function, if not provided will try to use `processor.__call__` or `tokenizer.__call__`
- `encode_kwargs` - the encoding function kwargs, if not provided will be `{"return_tensors": "pt"}`
- `generate` - the generation function, if not provided will try to use `model.generate`
- `generate_kwargs` - the generation function kwargs, if not provided will be empty
- `apply_chat_template` - the chat template function, if not provided will use `processor.tokenizer.apply_chat_template` or `tokenizer.apply_chat_template`.  this is actually really important for generations for AgentBench to even work and likely the default template for the model that does not have this will result in the model failing almost all tasks
- `apply_chat_template_kwargs` - the chat template function kwargs, if not provided will be `{"tokenize": False, "add_generation_prompt": True}`


# Similar projects or alternatives (and why they didnt work for my usage)
- https://github.com/jquesnelle/transformers-openai-api
  - doesnt have `chat/completions` route
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
  - doesnt support most models that I need, to add a new model requires https://docs.vllm.ai/en/latest/models/adding_model.html which means forking the repo and patching a lot of stuff, not easy to inspect the model during inference to tell what is happening which is helpful for multimodal models
- https://github.com/lhenault/simpleAI/blob/main/src/simple_ai/api_models.py
  - actually might be pretty similar to what i need but found too late and adding another model is more burdensome than I would like
- https://github.com/lm-sys/FastChat/blob/main/playground/FastChat_API_GoogleColab.ipynb
  - requires spinning up 3 services to work, the convo templates make adding another model not super easy https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py


## Using FastChat approach:

Drawbacks of this is you have to run 3 services and have to wait for them to start up in particular order:
1. controller: `python -m fastchat.serve.controller --host=0.0.0.0`
2. model `python -m fastchat.serve.model_worker --model-path "adept/fuyu-8b" --model-names "adept/fuyu-8b" --host=0.0.0.0`
   1. can use `--load-8bit` but doesnt seem to improve speed or memory usage?
3. openapi `python -m fastchat.serve.openai_api_server --host=0.0.0.0 --port=11434`


```bash
MODEL_NAME=...
time curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "'"$MODEL_NAME"'", "max_tokens": 50, "messages": [{"role": "user", "content": "Hello!"}]}'
{"id":"chatcmpl-L6sdTmvpy3zqE8hVUG2uKM","object":"chat.completion","created":1718830977,"model":"...","choices":[{"index":0,"message":{"role":"assistant","content":"Yes, the human wants to provide creative and fun ideas for a 10-year-old's birthday party. What do you think would be the best idea for a 10-year-old?\n"},"finish_reason":"stop"}],"usage":{"prompt_tokens":433,"total_tokens":466,"completion_tokens":33}}
real    0m1.327s
user    0m0.004s
sys     0m0.009s
```

# Later todo
- [ ] allow serving multiple instances of model for concurrent requests
  - not clear how best to do this, easiest seems to be using https://docs.ray.io/en/latest/serve/model-multiplexing.html
  - alternative is just use `multiprocessing.Queue` so that can load 1 model on each gpu and serve from available queue.  have a feeling this will be more complicated than expected