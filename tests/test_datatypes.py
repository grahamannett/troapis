import unittest

from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionRequest,
    CompletionResponse,
    LogProb,
    LogProbToken,
    Message,
    Usage,
)


class TestDatatypes(unittest.TestCase):
    def test_log_prob_token(self):
        log_prob_token = LogProbToken(token="test", logprob=0.5)
        self.assertEqual(log_prob_token.token, "test")
        self.assertEqual(log_prob_token.logprob, 0.5)
        self.assertIsNone(log_prob_token.bytes)

    def test_log_prob(self):
        log_prob = LogProb(content=[])
        self.assertEqual(log_prob.content, [])

    def test_message(self):
        message = Message(role="system")
        self.assertEqual(message.role, "system")
        self.assertIsNone(message.content)
        self.assertIsNone(message.tool_calls)
        self.assertIsNone(message.logprobs)

    def test_choice(self):
        message = Message(role="system")
        choice = Choice(index=1, message=message, finish_reason="stop")
        self.assertEqual(choice.index, 1)
        self.assertEqual(choice.message, message)
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIsNone(choice.logprobs)

    def test_usage(self):
        usage = Usage(completion_tokens=10, prompt_tokens=5, total_tokens=15)
        self.assertEqual(usage.completion_tokens, 10)
        self.assertEqual(usage.prompt_tokens, 5)
        self.assertEqual(usage.total_tokens, 15)

    @unittest.skip("Worry about Completion(Legacy) later")
    def test_completion_request(self):
        completion_request = CompletionRequest(model="gpt-3", prompt="Hello, world!")
        self.assertEqual(completion_request.model, "gpt-3")
        self.assertEqual(completion_request.prompt, "Hello, world!")
        self.assertEqual(completion_request.max_tokens, 7)
        self.assertEqual(completion_request.temperature, 1.0)
        self.assertEqual(completion_request.top_p, 1.0)
        self.assertEqual(completion_request.n, 1)
        self.assertFalse(completion_request.stream)
        self.assertEqual(completion_request.logprobs, 0)
        self.assertFalse(completion_request.echo)
        self.assertIsNone(completion_request.stop)
        self.assertEqual(completion_request.presence_penalty, 0.0)
        self.assertEqual(completion_request.frequence_penalty, 0.0)
        self.assertEqual(completion_request.best_of, 0)
        self.assertIsNone(completion_request.logit_bias)
        self.assertEqual(completion_request.user, "")

    @unittest.skip("Worry about Completion(Legacy) later")
    def test_completion_response(self):
        completion_response = CompletionResponse(
            id="1", created=123456789, model="gpt-3"
        )
        self.assertEqual(completion_response.id, "1")
        self.assertEqual(completion_response.created, 123456789)
        self.assertEqual(completion_response.model, "gpt-3")
        self.assertEqual(completion_response.object, "text_completion")
        self.assertIsNone(completion_response.choices)
        self.assertIsNone(completion_response.usage)

    def test_chat_completion_request(self):
        chat_completion_request = ChatCompletionRequest(messages=[], model="gpt-3")
        self.assertEqual(chat_completion_request.messages, [])
        self.assertEqual(chat_completion_request.model, "gpt-3")
        self.assertEqual(chat_completion_request.frequency_penalty, 0.0)
        self.assertEqual(chat_completion_request.logit_bias, {})
        self.assertFalse(chat_completion_request.logprobs)
        self.assertIsNone(chat_completion_request.top_logprobs)
        self.assertIsNone(chat_completion_request.max_tokens)
        self.assertEqual(chat_completion_request.n, 1)
        self.assertEqual(chat_completion_request.presence_penalty, 0.0)
        self.assertIsNone(chat_completion_request.response_format)
        self.assertIsNone(chat_completion_request.seed)
        self.assertIsNone(chat_completion_request.service_tier)
        self.assertIsNone(chat_completion_request.stop)
        self.assertFalse(chat_completion_request.stream)
        self.assertIsNone(chat_completion_request.stream_options)
        self.assertEqual(chat_completion_request.temperature, 1.0)
        self.assertEqual(chat_completion_request.top_p, 1.0)
        self.assertIsNone(chat_completion_request.tools)
        self.assertIsNone(chat_completion_request.tool_choice)
        self.assertTrue(chat_completion_request.parallel_tool_calls)
        self.assertIsNone(chat_completion_request.user)

    def test_chat_completion_response(self):
        chat_completion_response = ChatCompletionResponse(
            id="1",
            choices=[],
            created=123456789,
            model="gpt-3",
            system_fingerprint="test",
        )
        self.assertEqual(chat_completion_response.id, "1")
        self.assertEqual(chat_completion_response.choices, [])
        self.assertEqual(chat_completion_response.created, 123456789)
        self.assertEqual(chat_completion_response.model, "gpt-3")
        self.assertEqual(chat_completion_response.system_fingerprint, "test")
        self.assertIsNone(chat_completion_response.usage)
        self.assertIsNone(chat_completion_response.service_tier)
        self.assertEqual(chat_completion_response.object, "chat.completion")


if __name__ == "__main__":
    unittest.main()
