import unittest

import transformers
from troapis.datatypes import CompletionInput, CompletionResponse, ModelInfo
from troapis.utils import generate_completion


class TestGenerateCompletion(unittest.TestCase):
    def setUp(self):
        model_name = "gpt2"
        self.completion_input = CompletionInput(
            model=model_name,
            prompt=["test"],
            max_tokens=10,
            temperature=1.0,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.model_info = ModelInfo(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
        )
        self.uid = "test_uid"

    def test_generate_completion(self):
        result = generate_completion(self.completion_input, self.model_info, self.uid)
        self.assertIsInstance(result, CompletionResponse)
        self.assertEqual(result.id, self.uid)
        self.assertEqual(result.model, self.model_info.model_name)
        self.assertIsInstance(result.created, int)
        self.assertIsInstance(result.choices, list)
        self.assertIsInstance(result.usage, dict)
