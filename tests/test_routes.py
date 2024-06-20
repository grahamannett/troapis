import unittest

from fastapi.testclient import TestClient

from troapis.app import make_app

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# not sure if there is a smaller model?
model_name = "gpt2"
device = "cuda"
model_config = AutoConfig.from_pretrained(model_name)
model_info = {
    "model_name": model_name,
    # marginally faster than from_pretrained since weights arent loaded
    "model": AutoModelForCausalLM.from_config(model_config),
    "tokenizer": AutoTokenizer.from_pretrained(model_name),
}


class TestRoutes(unittest.IsolatedAsyncioTestCase):
    def test_health(self):
        app = make_app(model_info_from=model_info)

        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_models(self):
        app = make_app(model_info_from=model_info)

        with TestClient(app) as client:
            response = client.get("/v1/models")
            assert response.status_code == 200
            assert response.json() == {"models": ["gpt2"]}

    def test_chat_completions(self):
        app = make_app(model_info_from=model_info)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt2",
                    "context": "Hello, my name is",
                    "max_tokens": 15,
                    "temperature": 0.7,
                },
            )
            assert response.status_code == 200
            assert "completion" in response.json()
