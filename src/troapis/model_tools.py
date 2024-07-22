from __future__ import annotations

import importlib
import os.path
import sys
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from troapis.constants import MODULE_NAME
from troapis import log

FuncWithArgs = tuple[callable, dict]


def _notnone(a):
    return a is not None


# available here as these are the likely the values you want to use unless model specific, in which case you can import these and then modify per model
_decode_kwargs = dict(skip_special_tokens=True)
_encode_kwargs = dict(add_special_tokens=False, return_tensors="pt")
_generate_kwargs = dict(do_sample=True)
_apply_chat_template_kwargs = dict(tokenize=False, add_generation_prompt=True)


@dataclass
class ModelInfo:
    model_name: str
    model: callable

    # optional
    processor: callable = None
    tokenizer: callable = None

    decode: callable = None
    decode_kwargs: dict = field(default_factory=_decode_kwargs.copy)

    encode: callable = None
    encode_kwargs: dict = field(default_factory=_encode_kwargs.copy)

    generate: callable = None
    generate_kwargs: dict = field(default_factory=_generate_kwargs.copy)

    apply_chat_template: callable = None
    apply_chat_template_kwargs: dict = field(default_factory=_apply_chat_template_kwargs.copy)

    end_of_text: list[str] | tuple[str] = None

    def __post_init__(self):
        self.accs = [v for v in (self.processor, self.tokenizer) if _notnone(v)]

        # if the processor is used, makes tokenizer easier to access
        if tokenizer := getattr(self.processor, "tokenizer", None):
            self.tokenizer = tokenizer

    @classmethod
    def from_filepath(cls, entrypoint: str) -> ModelInfo:
        if os.path.isfile(entrypoint):
            spec = importlib.util.spec_from_file_location(MODULE_NAME, entrypoint)
            model_entrypoint = importlib.util.module_from_spec(spec)
            sys.modules[MODULE_NAME] = model_entrypoint
            spec.loader.exec_module(model_entrypoint)
            model_info = model_entrypoint.model_info
        else:
            raise ModuleNotFoundError("didnt find model_hentrypoint.py")

        # allow for model_info to either be ModelInfo dataclass or dict
        if isinstance(model_info, dict):
            model_info = cls(**model_info)

        log.info(f"loaded: `{model_info.model_name}` from `{entrypoint}`")
        return model_info

    @property
    def device(self) -> str:
        return self.model.device

    def _get_func_from(
        self,
        name: str,
        subname: str = None,
        accs: list[str] = None,
        direct: tuple[object, str] = None,
    ):
        subname = subname or name  # if subname is None, then it is the same as name
        accs = accs or self.accs  # if accs is None, then try default accs

        if direct and (obj := getattr(*direct, None)) is not None:
            if (func := getattr(obj, name, None)) is not None:
                return func

        if (func := getattr(self, name, None)) is not None:
            return func

        for acc in filter(_notnone, accs):
            return getattr(acc, subname)

        raise ValueError(f"No {name} found")

    def get_dec(self) -> FuncWithArgs:
        return self._get_func_from("decode"), self.decode_kwargs

    def get_enc(self) -> FuncWithArgs:
        return self._get_func_from("encode", "__call__"), self.encode_kwargs

    def get_gen(self) -> FuncWithArgs:
        return self._get_func_from("generate", accs=[self.model]), self.generate_kwargs or {}

    def get_chat(self) -> FuncWithArgs:
        # the apply_chat_template is on the tokenizer in general despite the processor having the method as well
        direct = (self.processor, "tokenizer")
        return self._get_func_from("apply_chat_template", direct=direct), self.apply_chat_template_kwargs


class ModelHolder:
    models: dict[str, ModelInfo] = {}

    @classmethod
    def add_model(
        cls,
        model_info: dict | ModelInfo,
        **kwargs,
    ):
        if isinstance(model_info, dict):
            model_info |= kwargs
            model_name = model_info["model_name"]

            if not any(k in kwargs for k in ("tokenizer", "processor")):
                kwargs["tokenizer"] = AutoTokenizer.from_pretrained(model_name)

            model_info = ModelInfo(**model_info)

        cls.models[model_info.model_name] = model_info

    def __getitem__(self, model_name: str | int) -> ModelInfo:
        if isinstance(model_name, int):
            model_name = list(self.models.keys())[model_name]

        return self.models[model_name]

    def list_models(self) -> list[str]:
        return list(self.models.keys())


ModelHolder = ModelHolder()
