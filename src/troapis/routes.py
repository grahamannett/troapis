from fastapi import APIRouter, HTTPException, Request

from troapis import log, utils
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
    request: Request,
) -> ChatCompletionResponse:
    request_uid = request.app.state.request_state.uid

    model_info = ModelHolder[completion.model]
    try:
        response = await utils.generate_chat_completion(completion, model_info, uid=request_uid)
        log.info(f"got response: {response}")
        return response
    except Exception as exception:
        log.error(f"Error in post_chat_completions: {exception}")
        raise HTTPException(status_code=500, detail=str(exception))


@router.post("/completions")
async def post_completions(
    completion: CompletionRequest,
) -> CompletionResponse:
    raise HTTPException(status_code=501, detail="Not implemented")
