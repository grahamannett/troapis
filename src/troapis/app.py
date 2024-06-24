from functools import partial

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from troapis import utils
from troapis.model_tools import ModelInfo
from troapis.routes import router


class RequestState:
    request_uid: int = 0

    @property
    def uid(self):
        self.request_uid += 1
        return str(self.request_uid)


def make_app(
    model_info_from: str | dict | ModelInfo = "entrypoint",
    add_midleware: bool = True,
    allow_credentials: bool = True,
    allow_origins: list[str] = ["*"],
    allow_methods: list[str] = ["*"],
    allow_headers: list[str] = ["*"],
    **kwargs,
):
    app = FastAPI(
        title="Text Generation API",
        lifespan=partial(utils.lifespan, load_from=model_info_from, **kwargs),
    )

    if add_midleware:
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=allow_credentials,
            allow_origins=allow_origins,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

    app.include_router(router, prefix="/v1")
    app.state.request_state = RequestState()

    return app


def run_app(model_info_from: dict | ModelInfo = None):
    import uvicorn

    args = utils.Args.from_args()
    app = make_app(
        model_info_from=model_info_from or args.load_from,
        allow_credentials=args.allow_credentials,
        allow_headers=args.allow_headers,
        allow_methods=args.allow_methods,
        allow_origins=args.allow_origins,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run_app()
