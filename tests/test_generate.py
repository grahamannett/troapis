import unittest
import torch

import transformers
from troapis.datatypes import ChatCompletionRequest, ChatCompletionResponse, Usage
from troapis.model_tools import ModelInfo
from troapis.utils import generate_chat_completion


class TestGenerateCompletion(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        model_name = "gpt2"
        device = "cuda:0"
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.model_info = ModelInfo(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
        )
        self.uid = "test_uid"

    async def test_generate_completion(self):
        completion_input: ChatCompletionRequest = torch.load("tests/fixtures/completion_request.pt")
        completion_input.max_tokens = 1024
        result = await generate_chat_completion(completion_input, self.model_info, self.uid)

        self.assertIsInstance(result, ChatCompletionResponse)
        self.assertEqual(result.id, self.uid)
        self.assertEqual(result.model, self.model_info.model_name)
        self.assertIsInstance(result.created, int)
        self.assertIsInstance(result.choices, list)
        self.assertIsInstance(result.usage, Usage)
