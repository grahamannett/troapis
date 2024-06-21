from dataclasses import field

from pydantic import BaseModel, field_validator


class LogProbToken(BaseModel):
    token: str
    logprob: float
    bytes: list = None  # This can be `None` or a list of integers


class LogProb(BaseModel):
    content: list[LogProbToken]


class Message(BaseModel):
    content: str | list[str] | list[dict]
    role: str


class Choice(BaseModel):
    index: int
    message: Message
    # log probs change for completion and chat completion
    logprobs: LogProb | dict = None
    finish_reason: str = "stop"


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


## ### ###
## ACTUAL REQUEST AND RESPONSE DATA CLASSES
class CompletionRequest(BaseModel):
    """
    for LEGACY endpoint.  Prefer  ChatCompletionInput
    """

    model: str
    prompt: str | list[str]  # = "<|endoftext|>"
    suffix: str = ""
    max_tokens: int = 7
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: int = 0
    echo: bool = False
    stop: str | list = None
    presence_penalty: float = 0.0
    frequence_penalty: float = 0.0
    best_of: int = 0
    logit_bias: None | dict = None
    user: str = ""

    @field_validator("prompt", mode="before")
    def prompt_to_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class CompletionResponse(BaseModel):
    """
    response for /v1/completions
    """

    id: str
    created: int
    model: str
    object: str = "text_completion"
    choices: list[Choice]
    usage: Usage = None


class ChatCompletionRequest(BaseModel):
    # for messages, using the actual Message class is problematic during the request phase as needs to be used with the chat_template_func, much easier to use a dict
    messages: list[dict]
    model: str
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] = field(default_factory=dict)
    logprobs: bool = False
    top_logprobs: int = None
    max_tokens: int = None
    n: int = 1
    presence_penalty: float = 0.0
    response_format: dict[str, str] = None
    seed: int = None
    service_tier: str = None
    stop: str | list[str] = None
    stream: bool = False
    stream_options: None | dict = None
    temperature: float = 1.0
    top_p: float = 1.0
    tools: list[dict] = None
    tool_choice: str | dict = None
    parallel_tool_calls: bool = True
    user: str = None


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list[Choice]  # List of Choice
    created: int
    model: str
    system_fingerprint: str = ""
    usage: Usage = None
    service_tier: str = None
    object: str = "chat.completion"
