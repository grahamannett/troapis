import argparse
import importlib
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI
from transformers import PreTrainedTokenizer

from troapis import log
from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    Message,
    Choice,
)
from troapis.model_tools import ModelHolder, ModelInfo


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
        parser.add_argument(
            "--allow-credentials", type=bool, default=cls.allow_credentials
        )

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


def _load_from_entrypoint(entrypoint: str = "model_entrypoint.py"):
    if os.path.isfile(entrypoint):
        # seems more robust than:
        #   allow_import_from_dir(); importlib.import_module(file.replace(".py", ""))
        module_name = "model_entrypoint"
        spec = importlib.util.spec_from_file_location(module_name, entrypoint)
        model_entrypoint = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model_entrypoint
        spec.loader.exec_module(model_entrypoint)

        model_info = model_entrypoint.model_info

        log.info(f"loaded: `{model_info['model_name']}` from `{entrypoint}`")
    else:
        raise ModuleNotFoundError("didnt find model_entrypoint.py")

    return model_info


@asynccontextmanager
async def lifespan(app: FastAPI, load_from: str = "entrypoint", **kwargs):
    # this seems like a kinda generic way that i can allow the models to be
    # loaded while the app would be easily installable/runnable from a single
    # command line entrypoint
    load_strategies = {
        "entrypoint": _load_from_entrypoint,
        "args": lambda: kwargs["model_info"],
    }

    if isinstance(load_from, str):
        log.info(f"loading model from: {load_from}")
        model_info = load_strategies[load_from]()
    elif isinstance(load_from, dict):
        model_info = load_from

    if model_info:
        ModelHolder.add_model(**model_info)

    app.state.models = ModelHolder

    yield


def check_messages(messages, completion_request, uid):
    if not isinstance(messages[0], dict):
        log.error("Not clear what the messages are.  Save the messages and check")
        torch.save(completion_request, f"out/completion{uid}-{int(time.time())}.pt")
        raise TypeError("If this is multiple messages sent at once, not handled yet")


get_messages = {
    ChatCompletionRequest: lambda x: x.messages,
    CompletionRequest: lambda x: x.prompt,
}


async def generate_chat_completion(
    completion_request: ChatCompletionRequest,
    model_info: ModelInfo,
    uid: str = "",
):
    """this is here as makes it easier for testing purposes

    Args:
        completion_request (CompletionInput): _description_
        model_info (ModelInfo): _description_
    """
    choices = []
    text_inputs = []
    inputs = []

    device = model_info.device
    chat_temp_func, chat_temp_func_kwargs = model_info.get_chat_templater()
    enc_func, enc_kwargs = model_info.get_enc()
    dec_func, dec_kwargs = model_info.get_dec()
    gen_func, gen_kwargs = model_info.get_gen(completion_request)

    # need to format the messages as they will not come in as just strings
    prompt = chat_temp_func(completion_request.messages, **chat_temp_func_kwargs)
    # should be similar to tokenizer(prompt, return_tensors="pt")
    input = enc_func(prompt, **enc_kwargs)
    inputs.append(input)
    text_inputs.append(prompt)

    # generate completions
    log.info(f"generating completions for {len(inputs)} prompts")
    prompt_tokens, completion_tokens = 0, 0
    for i, input in enumerate(inputs):
        input = input.to(device)

        for ii in range(completion_request.n):
            # should be similar to model.generate(input, **gen_kwargs)
            output = gen_func(**input, **gen_kwargs)
            output = output.squeeze(0)

            # echo not on ChatCompletionRequest, not sure if there is equivalent
            if not getattr(completion_request, "echo", False):
                output = output[input.input_ids.shape[1] :]

            # should be similar to tokenizer.decode(output, **dec_kwargs)
            text = dec_func(output, **dec_kwargs)

            choices.append(
                Choice(
                    index=(i * completion_request.n) + ii,
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

    return ChatCompletionResponse(
        id=uid,
        model=model_info.model_name,
        created=int(time.time()),
        choices=choices,
        usage=usage_info,
    )
