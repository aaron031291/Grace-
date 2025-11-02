"""
Hunter Protocol API Endpoints
============================

REST API for Grace Hunter Protocol ingestion pipeline.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Import Hunter Pipeline
try:
    from grace.hunter.pipeline import HunterPipeline, IngestionContext
    HUNTER_AVAILABLE = True
except ImportError:
    logger.warning("Hunter Pipeline not available")
    HUNTER_AVAILABLE = False

# Global pipeline instance
_hunter_pipeline: Optional[HunterPipeline] = None


def get_hunter_pipeline() -> HunterPipeline:
    """Get or create Hunter pipeline instance"""
    global _hunter_pipeline
    if _hunter_pipeline is None:
        from grace.events import get_event_bus
        from grace.governance import get_governance_kernel
        
        event_bus = get_event_bus()
        governance = get_governance_kernel()
        
        _hunter_pipeline = HunterPipeline(
            event_bus=event_bus,
            governance_kernel=governance
        )
    return _hunter_pipeline


class SubmitModuleRequest(BaseModel):
    """Request to submit a module"""
    name: str
    version: str
    owner: str
    type: str = "code"
    code: Optional[str] = None


class ModuleStatusResponse(BaseModel):
    """Module status response"""
    correlation_id: str
    module_id: Optional[str]
    current_stage: str
    completed_stages: list
    trust_score: float
    quality_score: float
    security_passed: bool
    governance_decision: str
    deployed: bool


@router.post("/submit")
async def submit_module(request: SubmitModuleRequest):
    """
    Submit a module for Hunter Protocol processing
    
    Processes through complete 17-stage validation pipeline.
    """
    if not HUNTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hunter Protocol not available")
    
    pipeline = get_hunter_pipeline()
    
    # Prepare data
    if request.code:
        raw_data = request.code.encode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Code is required")
    
    metadata = {
        "name": request.name,
        "version": request.version,
        "owner": request.owner,
        "type": request.type
    }
    
    # Process through pipeline
    context = await pipeline.process(raw_data, metadata)
    
    # Return result
    return {
        "correlation_id": context.correlation_id,
        "module_id": context.module_id,
        "status": "deployed" if context.final_validation_passed else "rejected",
        "trust_score": context.trust_score,
        "quality_score": context.quality_score,
        "stages_completed": len(context.completed_stages),
        "governance_decision": context.governance_decision,
        "errors": context.errors,
        "warnings": context.warnings
    }


@router.post("/submit/file")
async def submit_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    version: str = Form(...),
    owner: str = Form(...),
    data_type: str = Form("document")
):
    """
    Submit a file for Hunter Protocol processing
    
    Supports: documents, code, media, structured data
    """
    if not HUNTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hunter Protocol not available")
    
    pipeline = get_hunter_pipeline()
    
    # Read file
    raw_data = await file.read()
    
    metadata = {
        "name": name,
        "version": version,
        "owner": owner,
        "type": data_type,
        "filename": file.filename,
        "mime_type": file.content_type
    }
    
    # Process through pipeline
    context = await pipeline.process(raw_data, metadata)
    
    return {
        "correlation_id": context.correlation_id,
        "module_id": context.module_id,
        "status": "deployed" if context.final_validation_passed else "rejected",
        "data_type": context.data_type.value if context.data_type else None,
        "trust_score": context.trust_score,
        "stages_completed": len(context.completed_stages),
        "endpoints": context.endpoints
    }


@router.get("/status/{correlation_id}")
async def get_status(correlation_id: str):
    """Get processing status for a correlation ID"""
    if not HUNTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hunter Protocol not available")
    
    pipeline = get_hunter_pipeline()
    
    # Check active contexts
    if correlation_id in pipeline.active_contexts:
        context = pipeline.active_contexts[correlation_id]
        status = "processing"
    else:
        # Check completed
        completed = [c for c in pipeline.completed if c.correlation_id == correlation_id]
        if completed:
            context = completed[0]
            status = "complete"
        else:
            raise HTTPException(status_code=404, detail="Correlation ID not found")
    
    return ModuleStatusResponse(
        correlation_id=context.correlation_id,
        module_id=context.module_id,
        current_stage=context.current_stage.value,
        completed_stages=context.completed_stages,
        trust_score=context.trust_score,
        quality_score=context.quality_score,
        security_passed=context.security_passed,
        governance_decision=context.governance_decision,
        deployed=context.final_validation_passed
    )


@router.get("/modules/{module_id}")
async def get_module_info(module_id: str):
    """Get information about a deployed module"""
    if not HUNTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hunter Protocol not available")
    
    pipeline = get_hunter_pipeline()
    
    # Find module in completed
    modules = [c for c in pipeline.completed if c.module_id == module_id]
    
    if not modules:
        raise HTTPException(status_code=404, detail="Module not found")
    
    context = modules[0]
    
    return {
        "module_id": module_id,
        "name": context.metadata.get("name"),
        "version": context.metadata.get("version"),
        "owner": context.metadata.get("owner"),
        "data_type": context.data_type.value if context.data_type else None,
        "trust_score": context.trust_score,
        "quality_score": context.quality_score,
        "deployed_at": context.deployed_at.isoformat() if context.deployed_at else None,
        "endpoints": context.endpoints,
        "monitoring_active": context.monitoring_active
    }


@router.get("/stats")
async def get_pipeline_stats():
    """Get Hunter Protocol pipeline statistics"""
    if not HUNTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hunter Protocol not available")
    
    pipeline = get_hunter_pipeline()
    
    return pipeline.get_stats()
