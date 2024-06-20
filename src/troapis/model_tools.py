from dataclasses import dataclass, field
from types import ModuleType


@dataclass
class ModelInfo:
    model_name: str
    model: callable

    # optional
    processor: callable = None
    tokenizer: callable = None

    dec_func: callable = None
    dec_kwargs: dict = field(default_factory={"skip_special_tokens": True}.copy)

    enc_func: callable = None
    enc_kwargs: dict = field(default_factory={"return_tensors": "pt"}.copy)

    gen_func: callable = None
    gen_kwargs: dict = None

    @property
    def device(self):
        return self.model.device

    def get_dec(self):
        if self.decode_func:
            func = self.decode_func
        elif self.processor:
            func = self.processor.decode
        elif self.tokenizer:
            func = self.tokenizer.decode
        else:
            raise ValueError("No decode function found")
        return func, self.dec_kwargs

    def get_enc(self):
        if self.enc_func:
            func = self.enc_func
        elif self.processor:
            func = self.processor
        elif self.tokenizer:
            func = self.tokenizer
        else:
            raise ValueError("No encode function found")
        return func, self.enc_kwargs

    def get_gen(self, completion_request):
        if self.gen_func:
            func = self.gen_func
        elif hasattr(self.model, "generate"):
            func = self.model.generate
        else:
            raise ValueError("No generate function found")
        return func, {
            **self.gen_kwargs,
            "max_length": completion_request.max_tokens,
            "temperature": completion_request.temperature,
        }


class ModelHolder:
    models: dict[str, ModelInfo] = {}  # {field(default_factory=dict)}

    @classmethod
    def add_model(cls, model_name: str, model: callable, **kwargs):
        cls.models[model_name] = ModelInfo(model_name=model_name, model=model, **kwargs)

    @staticmethod
    def marshall_from_file(mod: ModuleType) -> ModelInfo:
        model_info = ModelInfo(
            model_name=mod.model_name,
            model=mod.model,
            processor=mod.processor,
        )
        return model_info

    def __getitem__(self, model_name: str | int) -> ModelInfo:
        if isinstance(model_name, int):
            model_name = list(self.models.keys())[model_name]

        return self.models[model_name]

    def list_models(self) -> list[str]:
        return list(self.models.keys())


ModelHolder = ModelHolder()


if __name__ == "__main__":
    mi = ModelInfo("gpt2", model=lambda x: x, enc_kwargs={"return_tensors": "pt"})
    print(mi)
