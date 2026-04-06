"""
Inference routes — Phase 4.
Provides /api/inference/* and /v1/* (OpenAI-compatible).
"""
import asyncio
import json
import logging
import time
import threading
import uuid
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

# Two routers — one for /api/inference, one for /v1 (OpenAI compat)
inference_router = APIRouter(prefix="/api/inference", tags=["inference"])
openai_router = APIRouter(prefix="/v1", tags=["openai-compat"])


# ── Request / Response models ──

class LoadRequest(BaseModel):
    model_path: str
    gguf_variant: Optional[str] = None
    hf_token: Optional[str] = None
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    chat_template_override: Optional[str] = None
    cache_type_kv: Optional[str] = None


class UnloadRequest(BaseModel):
    model_path: Optional[str] = None


class ValidateRequest(BaseModel):
    model_path: str
    hf_token: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: Any  # str or list[dict] for vision


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float = 0.7          # 0 = greedy decoding, >0 = sampling
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    max_tokens: int = 512
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    system: Optional[str] = None      # system prompt override
    enable_thinking: bool = False
    use_adapter: Optional[bool] = None   # None=default, False=base model, True=LoRA
    compare_slot: bool = False           # True = route to right-panel compare manager


# ── SSE helpers ──

def _make_chunk(content: str, model: str, finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_error_chunk(error: str, model: str) -> str:
    # Return error as chat content so the frontend displays it instead of silently dropping it
    chunk = {
        "id": "chatcmpl-error",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": f"⚠️ Generation error: {error}"},
            "finish_reason": "error",
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ── /api/inference routes ──

@inference_router.post("/load")
async def load_model(request: LoadRequest, _user=Depends(get_current_user)):
    """Load a model for inference. Auto-detects GGUF vs HF."""
    from core.inference import get_inference_manager
    manager = get_inference_manager()

    if manager.is_loading:
        return {"status": "already_loading", "model_name": manager.loading_model}

    try:
        result = await manager.load(
            model_path=request.model_path,
            gguf_variant=request.gguf_variant,
            hf_token=request.hf_token or None,
            max_seq_length=request.max_seq_length,
            load_in_4bit=request.load_in_4bit,
            chat_template_override=request.chat_template_override,
            cache_type_kv=request.cache_type_kv,
        )
        if result.get("status") == "error":
            raise HTTPException(500, result.get("error", "Load failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading model: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to load model: {e}")


@inference_router.post("/unload")
def unload_model(_request: UnloadRequest = UnloadRequest(), _user=Depends(get_current_user)):
    from core.inference import get_inference_manager
    return get_inference_manager().unload()


# ── Compare slot (right panel) — independent second model in VRAM ──

@inference_router.post("/load-compare")
async def load_compare_model(request: LoadRequest, _user=Depends(get_current_user)):
    """Load the right-panel model for compare mode into a separate subprocess."""
    from core.inference import get_compare_manager
    manager = get_compare_manager()
    if manager.is_loading:
        return {"status": "already_loading", "model_name": manager.loading_model}
    try:
        result = await manager.load(
            model_path=request.model_path,
            gguf_variant=request.gguf_variant,
            hf_token=request.hf_token or None,
            max_seq_length=request.max_seq_length,
            load_in_4bit=request.load_in_4bit,
            chat_template_override=request.chat_template_override,
            cache_type_kv=request.cache_type_kv,
        )
        if result.get("status") == "error":
            raise HTTPException(500, result.get("error", "Load failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading compare model: %s", e, exc_info=True)
        raise HTTPException(500, f"Failed to load compare model: {e}")


@inference_router.post("/unload-compare")
def unload_compare_model(_user=Depends(get_current_user)):
    from core.inference import get_compare_manager
    return get_compare_manager().unload()


@inference_router.get("/status-compare")
def compare_model_status(_user=Depends(get_current_user)):
    from core.inference import get_compare_manager
    m = get_compare_manager()
    return {
        "is_loaded": m.is_model_loaded(),
        "is_loading": m.is_loading,
        "model_name": m.loaded_model_name or m.loading_model or None,
    }


@inference_router.post("/validate")
def validate_model(request: ValidateRequest, _user=Depends(get_current_user)):
    """Validate a model path without loading it."""
    path = request.model_path
    lower = path.lower()
    is_gguf = lower.endswith(".gguf") or "gguf" in lower
    is_local = __import__("pathlib").Path(path).exists()
    return {
        "valid": True,
        "model_path": path,
        "is_gguf": is_gguf,
        "is_local": is_local,
        "is_vision": any(x in lower for x in ["vl", "vision", "llava"]),
    }


@inference_router.get("/status")
def inference_status(_user=Depends(get_current_user)):
    from core.inference import get_inference_manager
    return get_inference_manager().get_status()


@inference_router.post("/chat/completions")
async def chat_completions_audio(request: ChatCompletionRequest, _user=Depends(get_current_user)):
    """Text-to-speech / audio endpoint (alias for standard completions)."""
    return await _handle_completions(request)


# ── /v1 OpenAI-compatible routes ──

@openai_router.get("/models")
def list_models_openai(_user=Depends(get_current_user)):
    """OpenAI-compatible model list."""
    from core.inference import get_inference_manager
    manager = get_inference_manager()
    models_data = []
    if manager.active_model:
        models_data.append({
            "id": manager.active_model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "unslothcraft",
        })
    # Add some well-known defaults
    for name in ["unsloth/Llama-3.2-3B-Instruct", "unsloth/Qwen2.5-7B-Instruct"]:
        if name != manager.active_model:
            models_data.append({"id": name, "object": "model", "created": 0, "owned_by": "unsloth"})
    return {"object": "list", "data": models_data}


@openai_router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request, _user=Depends(get_current_user)):
    """OpenAI-compatible chat completions endpoint with SSE streaming."""
    return await _handle_completions(request)


async def _handle_completions(request: ChatCompletionRequest):
    """Shared handler for streaming chat completions.
    Routes to compare manager when request.compare_slot=True."""
    from core.inference import get_inference_manager, get_compare_manager
    manager = get_compare_manager() if request.compare_slot else get_inference_manager()

    if not manager.is_loaded:
        slot = "compare (right)" if request.compare_slot else "primary (left)"
        raise HTTPException(400, f"No model loaded in {slot} slot. Load a model first.")

    model_name = manager.active_model or request.model

    # Convert messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    cancel_event = threading.Event()

    if request.stream:
        async def event_stream():
            try:
                # Run blocking generator in a thread
                loop = asyncio.get_event_loop()
                token_queue: asyncio.Queue = asyncio.Queue()

                def _generate():
                    try:
                        gen = manager.stream_completions(
                            messages=messages,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            top_k=request.top_k,
                            min_p=request.min_p,
                            max_tokens=request.max_tokens,
                            repetition_penalty=request.repetition_penalty,
                            presence_penalty=request.presence_penalty,
                            system_prompt=request.system,
                            enable_thinking=request.enable_thinking,
                            use_adapter=request.use_adapter,
                            cancel_event=cancel_event,
                        )
                        for token in gen:
                            loop.call_soon_threadsafe(token_queue.put_nowait, token)
                    except Exception as e:
                        loop.call_soon_threadsafe(token_queue.put_nowait, Exception(str(e)))
                    finally:
                        loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

                import concurrent.futures
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = loop.run_in_executor(executor, _generate)

                # Stream tokens as SSE
                while True:
                    # Longer timeout for first token (model may need to load/warm up)
                    token = await asyncio.wait_for(token_queue.get(), timeout=120.0)
                    if token is None:
                        break
                    if isinstance(token, Exception):
                        yield _make_error_chunk(str(token), model_name)
                        break
                    if token:
                        yield _make_chunk(token, model_name)

                yield _make_chunk("", model_name, finish_reason="stop")
                yield "data: [DONE]\n\n"

                await future

            except asyncio.TimeoutError:
                cancel_event.set()
                yield _make_error_chunk("Generation timed out", model_name)
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("Stream error: %s", e, exc_info=True)
                yield _make_error_chunk(str(e), model_name)
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming: collect all tokens then return
        def _collect():
            return "".join(manager.stream_completions(
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                system_prompt=request.system,
                cancel_event=cancel_event,
            ))

        loop = asyncio.get_event_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            content = await loop.run_in_executor(ex, _collect)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        }
