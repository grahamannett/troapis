from dataclasses import dataclass, field

from transformers import AutoTokenizer

FuncWithArgs = tuple[callable, dict]


@dataclass
class ModelInfo:
    model_name: str
    model: callable

    # optional
    processor: callable = None
    tokenizer: callable = None

    decode: callable = None
    decode_kwargs: dict = field(default_factory={"skip_special_tokens": True}.copy)

    encode: callable = None
    encode_kwargs: dict = field(default_factory={"return_tensors": "pt"}.copy)

    generate: callable = None
    generate_kwargs: dict = None

    apply_chat_template: callable = None
    apply_chat_template_kwargs: dict = field(
        default_factory={"tokenize": False, "add_generation_prompt": True}.copy,
    )

    @property
    def device(self) -> str:
        return self.model.device

    def _get_func_from(
        self,
        name: str,
        subname: str = None,
        accs: list[str] = None,
    ):
        subname = subname or name  # if subname is None, then it is the same as name
        accs = accs or [self.processor, self.tokenizer]  # if accs is None, then it is the default, nicer than

        def _notnone(a):
            return a is not None

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
        return self._get_func_from("apply_chat_template"), self.apply_chat_template_kwargs


class ModelHolder:
    models: dict[str, ModelInfo] = {}

    @classmethod
    def add_model(cls, model_name: str, model: callable, **kwargs):
        if "tokenizer" and "processor" not in kwargs:
            kwargs["tokenizer"] = AutoTokenizer.from_pretrained(model_name)

        cls.models[model_name] = ModelInfo(model_name=model_name, model=model, **kwargs)

    def __getitem__(self, model_name: str | int) -> ModelInfo:
        if isinstance(model_name, int):
            model_name = list(self.models.keys())[model_name]

        return self.models[model_name]

    def list_models(self) -> list[str]:
        return list(self.models.keys())


ModelHolder = ModelHolder()
