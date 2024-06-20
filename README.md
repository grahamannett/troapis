# troapis

I am not sure why there isnt something like this already, basically a bare bones way to serve dev model from openai api specifically for benchmarks like AgentBench which need it to be so

# Usage

ways to use this:

one is via just writing a model_entrypoint.py file that loads the model in the file, the server will then check for that file and try and load it

can also run the app and pass it in (as seen in model_entrypoint `if __name__ == "__main__":` part)

```python
from troapis.app import run_app
model_info = load_model()
run_app(model_info=model_info)
```

to use dev can just do `pdm run dev`

# Similar projects
- https://github.com/jquesnelle/transformers-openai-api
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
- https://github.com/lhenault/simpleAI/blob/main/src/simple_ai/api_models.py
- https://github.com/lm-sys/FastChat/blob/main/playground/FastChat_API_GoogleColab.ipynb


# Using FastChat approach:

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