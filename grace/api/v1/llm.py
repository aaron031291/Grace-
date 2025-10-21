"""
LLM API endpoints - Private local model inference
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from grace.auth.dependencies import get_current_user
from grace.auth.models import User
from grace.llm import ModelManager, InferenceRouter, ModelConfig, LLMProvider

router = APIRouter(prefix="/llm", tags=["Private LLM"])

# Global model manager (initialized on startup)
model_manager: Optional[ModelManager] = None
inference_router: Optional[InferenceRouter] = None


def get_inference_router() -> InferenceRouter:
    """Get inference router dependency"""
    if inference_router is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return inference_router


class GenerateRequest(BaseModel):
    """Text generation request"""
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = None
    task_type: str = Field("general", description="general, code, complex")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = False


class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="system, user, or assistant")
    content: str


class ChatRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: bool = False


@router.post("/generate")
async def generate_text(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user),
    router: InferenceRouter = Depends(get_inference_router)
):
    """
    Generate text from prompt using local LLM
    
    No external API calls - completely private
    """
    try:
        result = router.route(
            prompt=request.prompt,
            task_type=request.task_type,
            preferred_model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream
        )
        
        return {
            "generated_text": result["text"],
            "tokens_generated": result["tokens"],
            "model_used": result["routed_to"],
            "finish_reason": result.get("finish_reason"),
            "provider": result.get("provider")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/chat")
async def chat_completion(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    router: InferenceRouter = Depends(get_inference_router)
):
    """
    Chat completion using local LLM
    
    Compatible with OpenAI chat format but 100% private
    """
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        result = router.chat(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": result.get("finish_reason")
            }],
            "model": result["routed_to"],
            "usage": {
                "completion_tokens": result["tokens"],
                "total_tokens": result["tokens"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/models")
async def list_models(
    current_user: User = Depends(get_current_user)
):
    """List available local models"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return {
        "models": model_manager.list_models()
    }


async def initialize_llm_service():
    """Initialize LLM service on startup"""
    global model_manager, inference_router
    
    model_manager = ModelManager()
    model_manager.load_default_models()
    
    inference_router = InferenceRouter(model_manager)
    
    logger.info("LLM service initialized")
