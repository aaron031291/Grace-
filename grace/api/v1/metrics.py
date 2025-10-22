"""
Metrics API endpoints
"""

from fastapi import APIRouter, Response
from grace.observability.metrics import get_metrics_collector

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/")
async def get_metrics_json():
    """Get metrics in JSON format"""
    collector = get_metrics_collector()
    metrics = await collector.get_metrics()
    return metrics


@router.get("/prometheus")
async def get_metrics_prometheus():
    """Get metrics in Prometheus format"""
    collector = get_metrics_collector()
    metrics_text = await collector.get_prometheus_metrics()
    return Response(content=metrics_text, media_type="text/plain")
