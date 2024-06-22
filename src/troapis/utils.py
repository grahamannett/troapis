import argparse
import importlib
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from fastapi import FastAPI

from troapis import log
from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
)
from troapis.model_tools import FuncWithArgs, ModelHolder, ModelInfo

DEBUG_MODE = os.environ.get("DEBUG", "false").lower() in ["true", "1"]
ENTRYPOINT_PATH = "model_entrypoint.py"
MODULE_NAME = "model_entrypoint"


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


def _load_from_entrypoint(entrypoint: str = None) -> ModelInfo | dict:
    if not entrypoint:
        entrypoint = os.environ.get("ENTRYPOINT", ENTRYPOINT_PATH)

    if os.path.isfile(entrypoint):
        # seems more robust than:
        #   allow_import_from_dir(); importlib.import_module(file.replace(".py", ""))
        spec = importlib.util.spec_from_file_location(MODULE_NAME, entrypoint)
        model_entrypoint = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = model_entrypoint
        spec.loader.exec_module(model_entrypoint)

        model_info = model_entrypoint.model_info
        # allow for model_info to either be ModelInfo dataclass or dict
        model_name = model_info.get("model_name", getattr(model_info, "model_name"))

        log.info(f"loaded: `{model_name}` from `{entrypoint}`")
    else:
        raise ModuleNotFoundError("didnt find model_hentrypoint.py")

    return model_info


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
        "entrypoint": _load_from_entrypoint,
        "args": lambda: kwargs["model_info"],
    }

    if isinstance(load_from, str):
        log.info(f"loading model from: {load_from}")
        model_info = load_strategies[load_from](**kwargs)
    elif isinstance(load_from, dict):
        model_info = load_from

    if model_info:
        ModelHolder.add_model(**model_info)

    app.state.models = ModelHolder

    yield


def _chat_completion_check(
    request: ChatCompletionRequest,
    uid: str | int,
    save: bool = DEBUG_MODE,
) -> None:
    if not isinstance(request.messages[0], dict):
        log.error(f"Not clear what the messages are: {request.messages}")

        if save:
            save_file = f"completion{uid}-{int(time.time())}.pt"
            log.debug(f"Saving request to out/{save_file}")
            torch.save(request, f"out/{save_file}")

        raise TypeError("messages should be a list of dictionaries")


async def generate_chat_completion(
    request: ChatCompletionRequest,
    model_info: ModelInfo,
    uid: str = "",
):
    """this is here as makes it easier for testing purposes

    Args:
        request (ChatCompletionRequest): _description_
        model_info (ModelInfo): _description_
    """
    choices = []
    inputs = []

    device = model_info.device
    chat_func, chat_func_kwargs = model_info.get_chat()
    enc_func, enc_kwargs = model_info.get_enc()
    dec_func, dec_kwargs = model_info.get_dec()
    gen_func, gen_kwargs = model_info.get_gen()

    gen_kwargs |= {"max_length": request.max_tokens, "temperature": request.temperature}

    # need to format the messages as they will not come in as just strings
    _chat_completion_check(request, uid, save=DEBUG_MODE)
    prompt = chat_func(request.messages, **chat_func_kwargs)
    # some templates have [INST] which will be problematic for rich
    log.debug("generate for prompt below\n---⤵️---\n"), log.debug(prompt, markup=False)

    # should be similar to tokenizer(prompt, return_tensors="pt")
    input = enc_func(prompt, **enc_kwargs)
    inputs.append(input)

    # generate completions
    log.debug(f"generating completions for {len(inputs)} prompts")
    prompt_tokens, completion_tokens = 0, 0
    for i, input in enumerate(inputs):
        input = input.to(device)

        for ii in range(request.n):
            # should be similar to model.generate(input, **gen_kwargs)
            output = gen_func(**input, **gen_kwargs)
            output = output.squeeze(0)

            # echo not on ChatCompletionRequest, not sure if there is equivalent
            if not getattr(request, "echo", False):
                output = output[input.input_ids.shape[1] :]

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

    return ChatCompletionResponse(
        id=uid,
        model=model_info.model_name,
        created=int(time.time()),
        choices=choices,
        usage=usage_info,
    )


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
