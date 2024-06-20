from fastapi import APIRouter, HTTPException

from troapis import utils
from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)

from troapis.model_tools import ModelHolder

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/models")
async def get_models():
    return {"models": ModelHolder.list_models()}


@router.post("/chat/completions")
async def post_chat_completions(
    completion: ChatCompletionRequest,
) -> ChatCompletionResponse:
    model_info = ModelHolder[completion.model]

    try:
        return utils.generate_completion(completion, model_info)
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))


@router.post("/completions")
async def post_completions(
    completion: CompletionRequest,
) -> CompletionResponse:
    model_info = ModelHolder[completion.model]
    try:
        return utils.generate_completion(completion, model_info)
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))
