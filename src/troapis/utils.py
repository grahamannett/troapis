import argparse
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI

from troapis import log
from troapis.constants import DEBUG_MODE
from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
)
from troapis.model_tools import FuncWithArgs, ModelHolder, ModelInfo


@dataclass
class Args:
    load_from: str = "entrypoint"

    # server settings
    host: str = "0.0.0.0"
    port: int = 11434
    # not sure but possibly these are necessary (at least allow_credentials)
    allow_credentials: bool = True

    allow_headers: list = field(default_factory=lambda: ["*"])
    allow_methods: list = field(default_factory=lambda: ["*"])
    allow_origins: list = field(default_factory=lambda: ["*"])

    @classmethod
    def from_args(cls):
        def fn(name):
            return cls.__dataclass_fields__[name].default_factory()

        parser = argparse.ArgumentParser()
        parser.add_argument("--load-from", type=str, default=cls.load_from)
        parser.add_argument("--host", type=str, default=cls.host)
        parser.add_argument("--port", type=int, default=cls.port)
        parser.add_argument("--allow-credentials", type=bool, default=cls.allow_credentials)

        parser.add_argument("--allow-headers", type=list, default=fn("allow_headers"))
        parser.add_argument("--allow-methods", type=list, default=fn("allow_methods"))
        parser.add_argument("--allow-origins", type=list, default=fn("allow_origins"))
        parsed_args = parser.parse_args()
        return cls(**vars(parsed_args))


def allow_import_from_dir(model_dir: str = None):
    from os import getcwd

    if model_dir is None:
        model_dir = getcwd()

    sys.path.append(model_dir)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    load_from: str | dict | ModelInfo = "entrypoint",
    **kwargs,
):
    # this seems like a kinda generic way that i can allow the models to be
    # loaded while the app would be easily installable/runnable from a single
    # command line entrypoint
    load_strategies = {
        "entrypoint": ModelInfo.from_filepath,
        "args": lambda: kwargs["model_info"],
    }

    if env_entrypoint := os.environ.get("ENTRYPOINT"):
        kwargs["entrypoint"] = env_entrypoint

    if isinstance(load_from, str):
        log.info(f"loading model from: {load_from}")
        model_info = load_strategies[load_from](**kwargs)
    elif isinstance(load_from, dict):
        model_info = load_from

    ModelHolder.add_model(model_info, **kwargs)

    app.state.models = ModelHolder

    yield


def _chat_completion_check(
    request: ChatCompletionRequest,
    uid: str | int,
    save: bool = DEBUG_MODE,
) -> None:
    if save:
        save_file = f"completion{uid}-{int(time.time())}.pt"
        log.info(f"Saving request to out/{save_file}")
        torch.save(request, f"out/{save_file}")

    if not isinstance(request.messages[0], dict):
        log.error(f"Not clear what the messages are: {request.messages}")
        raise TypeError("messages should be a list of dictionaries")


async def generate_chat_completion(
    request: ChatCompletionRequest,
    model_info: ModelInfo,
    uid: str = "",
    end_of_text: list | tuple = (),
    done_callback: callable = None,
    # override specific flags/kwargs
    override_echo: bool = None,
):
    """Generates chat completions based on the given request and model information.

    Args:
        request (ChatCompletionRequest): The request object containing the chat completion parameters.
        model_info (ModelInfo): The model information object containing the necessary functions and arguments.
        uid (str, optional): The unique identifier for the chat completion. Defaults to "".
        echo_override (bool, optional): Overrides the echo setting in the request. Defaults to None.
        done_callback (callable, optional): The callback function to be called when the chat completion is done. Defaults to None.  Useful for logging additional/saving the request/response

    Returns:
        ChatCompletionResponse: The response object containing the generated chat completions.

    """
    choices = []
    inputs = []

    device = model_info.device
    chat_func, chat_func_kwargs = model_info.get_chat()
    enc_func, enc_kwargs = model_info.get_enc()
    dec_func, dec_kwargs = model_info.get_dec()
    gen_func, gen_kwargs = model_info.get_gen()

    gen_kwargs |= {"max_new_tokens": request.max_tokens, "temperature": request.temperature}

    # echo not on ChatCompletionRequest, not sure if there is equivalent in ChatCompletionRequest but allow override
    # for calling `generate_chat_completion` independently
    echo_input = getattr(request, "echo", False) if (override_echo is None) else override_echo

    # need to format the messages as they will not come in as just strings
    _chat_completion_check(request, uid, save=DEBUG_MODE)
    prompt = chat_func(request.messages, **chat_func_kwargs)
    # some templates have [INST] which will be problematic for rich
    # should be similar to tokenizer(prompt, return_tensors="pt")
    input = enc_func(prompt, **enc_kwargs)
    inputs.append(input)

    # generate completions
    log.rule(title=f"Generating from Prompt⤵️|UID:{uid}|TEMP:{request.temperature}")
    log.with_escape(log.info, prompt)

    prompt_tokens, completion_tokens = 0, 0
    for i, input in enumerate(inputs):
        input = input.to(device)

        for ii in range(request.n):
            # should be similar to model.generate(input, **gen_kwargs)

            output = gen_func(**input, **gen_kwargs)
            output = output.squeeze(0)

            if not echo_input:
                output = output[input.input_ids.shape[1] :]

            for item in end_of_text:
                if item in output:
                    output = output.split(item)[0]

            # should be similar to tokenizer.decode(output, **dec_kwargs)
            text = dec_func(output, **dec_kwargs)

            choices.append(
                Choice(
                    index=(i * request.n) + ii,
                    message=Message(
                        role="assistant",
                        content=text,
                    ),
                )
            )

            prompt_tokens += input.input_ids.shape[-1]
            completion_tokens += output.shape[-1]

    usage_info = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    response = ChatCompletionResponse(
        id=uid,
        model=model_info.model_name,
        created=int(time.time()),
        choices=choices,
        usage=usage_info,
    )

    if done_callback:
        done_callback(**locals())

    return response


async def generate_completion(request: CompletionRequest, model_info: ModelInfo, uid: str = "") -> CompletionResponse:
    # not going to implement unless i see where this is needed in a benchmark. ideally would just be one function for
    # this and generate_chat_completion but they have different formats and return types
    raise NotImplementedError("generate_completion not implemented")


def _generate(
    prompt: str,
    enc: FuncWithArgs,
    dec: FuncWithArgs,
    gen: FuncWithArgs,
    device: str,
    echo: bool = False,
    num_gen: int = 1,
):
    # should migrate to using generic generate between all generates if
    #  implementing the generate_completion
    enc_func, enc_kwargs = enc
    dec_func, dec_kwargs = dec
    gen_func, gen_kwargs = gen

    inputs = [enc_func(prompt, **enc_kwargs)]

    # generate completions
    generations = []
    prompt_tokens, completion_tokens = 0, 0

    for inp in inputs:
        inp = inp.to(device)

        for _ in num_gen:
            # should be similar to model.generate(inp, **gen_kwargs)
            output = gen_func(**inp, **gen_kwargs)
            output = output.squeeze(0)

            # echo not on ChatCompletionRequest, not sure if there is equivalent
            if not echo:
                output = output[input.input_ids.shape[1] :]

            # should be similar to tokenizer.decode(output, **dec_kwargs)
            text = dec_func(output, **dec_kwargs)
            generations.append(text)

            prompt_tokens += input.input_ids.shape[-1]
            completion_tokens += output.shape[-1]

    usage_info = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return generations, usage_info
