import argparse
import importlib
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI

from troapis import log
from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
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
        parser = argparse.ArgumentParser()
        parser.add_argument("--load-from", type=str, default=cls.load_from)
        parser.add_argument("--host", type=str, default=cls.host)
        parser.add_argument("--port", type=int, default=cls.port)
        parser.add_argument(
            "--allow-credentials", type=bool, default=cls.allow_credentials
        )
        parser.add_argument("--allow-headers", type=list, default=cls.allow_headers())
        parser.add_argument("--allow-methods", type=list, default=cls.allow_methods())
        parser.add_argument("--allow-origins", type=list, default=cls.allow_origins())
        args = parser.parse_args()
        return cls(**vars(args))


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

        log.info(f"loaded: `{model_entrypoint.model_name}` from `{entrypoint}`")
    else:
        raise ModuleNotFoundError("didnt find model_entrypoint.py")

    return model_entrypoint.ModelInfo


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


def generate_completion(
    completion_request: CompletionRequest | ChatCompletionRequest,
    model_info: ModelInfo,
    uid: str = "",
    decode_kwargs: dict = {"skip_special_tokens": True},
):
    """this is here as makes it easier for testing purposes

    Args:
        completion_request (CompletionInput): _description_
        model_info (ModelInfo): _description_
    """
    choices = []
    inputs = []

    device = model_info.device
    enc_func, enc_kwargs = model_info.get_encode()
    dec_func, dec_kwargs = model_info.get_decode()
    gen_func, gen_kwargs = model_info.get_generate(completion_request)

    for prompt in completion_request.prompt:
        # should be similar to tokenizer(prompt, return_tensors="pt")
        input = enc_func(prompt, **enc_kwargs)
        inputs.append(input)

    # generate completions
    log.info(f"generating completions for {len(inputs)} prompts")
    for i, input in enumerate(inputs):
        input = input.to(device)

        for _ in range(completion_request.n):
            # should be similar to model.generate(input, **gen_kwargs)
            output = gen_func(**input, **gen_kwargs)
            output = output.squeeze(0)

            if completion_request.echo is False:
                output = output[input.input_ids.shape[1] :]

            # should be similar to tokenizer.decode(output, **dec_kwargs)
            text = dec_func(output, **dec_kwargs)

            choices.append(
                {
                    "text": text,
                    "index": i,
                }
            )

    usage_info = {
        "prompt_tokens": sum(input.input_ids.size(dim=1) for input in inputs),
        "completion_tokens": sum(len(choice["text"]) for choice in choices),
    }

    return CompletionResponse(
        id=uid,
        model=model_info.model_name,
        created=int(time.time()),
        choices=choices,
        usage=usage_info,
    )
