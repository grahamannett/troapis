from fastapi import APIRouter, HTTPException, Request
from functools import cache
from pathlib import Path

from troapis import log, utils
from troapis.datatypes import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)
from troapis.model_tools import ModelHolder
from troapis.constants import REQ_SAVE_DIR, SAVE_MODE
import json

router = APIRouter()


@cache
def _make_request_filepath(model_name: str, uid: str, save_dir: str = REQ_SAVE_DIR):
    save_dir = f"{save_dir}/{model_name.replace('/', '_')}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return f"{save_dir}/{uid}.json"


async def _save_request(request: Request, model_name: str, uid: str, save_dir: str = REQ_SAVE_DIR):
    filepath = _make_request_filepath(model_name, uid, save_dir=save_dir)
    data = await request.json()

    # Assuming `data` is the dictionary you want to dump
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)
    log.info(f"Saved request-{uid} to {filepath}")


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
    request_filepath: str = "Not-Saved",
) -> ChatCompletionResponse:
    uid = request.app.state.request_state.uid
    model_info = ModelHolder[completion.model]

    if SAVE_MODE:
        request_filepath = await _save_request(request, model_info.name, uid)

    try:
        response = await utils.generate_chat_completion(completion, model_info, uid=uid)
        log.info(f"got response: {response} and saved request to {request_filepath}")
        return response
    except Exception as exception:
        log.error(f"Error in post_chat_completions: {exception}")
        raise HTTPException(status_code=500, detail=str(exception))


@router.post("/completions")
async def post_completions(
    completion: CompletionRequest,
) -> CompletionResponse:
    raise HTTPException(status_code=501, detail="Not implemented")
