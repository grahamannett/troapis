import unittest

import torch
import transformers
from troapis.datatypes import ChatCompletionRequest, ChatCompletionResponse, Usage
from troapis.model_tools import ModelInfo
from troapis.utils import generate_chat_completion

uid = "test_uid"
device = "cuda:0"

completion_request_fixture = "tests/fixtures/completion_request.pt"


class TestGenerateCompletion(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        model_name = "gpt2"

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.model_info = ModelInfo(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
        )

    async def test_generate_completion(self):
        completion_input: ChatCompletionRequest = torch.load(completion_request_fixture)
        completion_input.max_tokens = 1024
        result = await generate_chat_completion(completion_input, self.model_info, self.uid)

        self.assertIsInstance(result, ChatCompletionResponse)
        self.assertEqual(result.id, uid)
        self.assertEqual(result.model, self.model_info.model_name)
        self.assertIsInstance(result.created, int)
        self.assertIsInstance(result.choices, list)
        self.assertIsInstance(result.usage, Usage)


class TestEntrypointChatCompletion(unittest.IsolatedAsyncioTestCase):
    async def test_entrypoint(self):
        model_info = ModelInfo.from_filepath("model_entrypoint.py")

        completion_input: ChatCompletionRequest = torch.load(completion_request_fixture)
        completion_input.max_tokens = 1024
        result = await generate_chat_completion(completion_input, model_info, uid)
        self.assertIsInstance(result, ChatCompletionResponse)

    async def test_paligemma(self):
        paligemma_entrypoint = "entrypoints/paligemma.py"
        model_info = ModelInfo.from_filepath(paligemma_entrypoint)

        completion_input: ChatCompletionRequest = torch.load(completion_request_fixture)
        completion_input.max_tokens = 64
        completion_input.temperature = 0.5

        # output is the local var in generate_chat_completion
        def capture_cb(response, output, **kwargs):
            # field needs to start with "_" to attach on pydantic model
            response._output = output

        result = await generate_chat_completion(completion_input, model_info, uid, done_callback=capture_cb)

        # allow the capture callback to save the output to verify the length, length of 1 means kwargs are wrong in gen/template/etc
        self.assertTrue(hasattr(result, "_output"))
        self.assertGreater(result._output.shape[-1], 1)
