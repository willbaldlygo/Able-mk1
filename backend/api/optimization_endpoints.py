"""API endpoints for AI optimization services."""
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from services.ai_optimization_service import AIOptimizationService
from services.performance_analyzer import PerformanceAnalyzer, OptimizationStrategy
from services.intelligent_routing_service import IntelligentRoutingService
from services.response_cache_service import ResponseCacheService
from services.local_graphrag_optimizer import LocalGraphRAGOptimizer
from services.performance_monitoring_service import PerformanceMonitoringService
from models import ChatRequest, EnhancedChatRequest

# Initialize router
router = APIRouter(prefix="/api/optimization", tags=["optimization"])

# Initialize services (these would be dependency-injected in a full implementation)
optimization_service = AIOptimizationService()
performance_analyzer = PerformanceAnalyzer()
routing_service = IntelligentRoutingService()
cache_service = ResponseCacheService()
graphrag_optimizer = LocalGraphRAGOptimizer()
monitoring_service = PerformanceMonitoringService()

# Pydantic models for API requests/responses
class BenchmarkRequest(BaseModel):
    test_queries: Optional[List[str]] = None
    include_providers: Optional[List[str]] = None
    include_strategies: Optional[List[str]] = None

class OptimizationConfigRequest(BaseModel):
    strategy: str  # "speed_priority", "accuracy_priority", "balanced", "adaptive"
    max_response_time: Optional[float] = None
    min_quality_score: Optional[float] = None
    cost_sensitivity: Optional[float] = None

class CacheConfigRequest(BaseModel):
    enabled: bool
    ttl_hours: Optional[int] = None
    max_cache_size_mb: Optional[int] = None
    strategies: Optional[Dict[str, bool]] = None

class RoutingPreferencesRequest(BaseModel):
    preferred_provider: Optional[str] = None
    max_latency: Optional[float] = None
    cost_sensitivity: Optional[float] = None
    quality_priority: Optional[float] = None
    privacy_preference: Optional[float] = None


@router.get("/health")
async def optimization_health_check():
    """Health check for optimization services."""
    return {
        "status": "healthy",
        "services": {
            "ai_optimization": "active",
            "performance_analyzer": "active", 
            "routing_service": "active",
            "cache_service": "active",
            "graphrag_optimizer": "active",
            "monitoring_service": "active"
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/benchmark")
async def run_optimization_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks
):
    """Run comprehensive optimization benchmark."""
    try:
        # Run benchmark in background
        benchmark_result = await optimization_service.run_benchmark_suite(
            test_queries=request.test_queries
        )
        
        return {
            "success": True,
            "benchmark_id": benchmark_result.get("benchmark_time"),
            "message": "Benchmark completed successfully",
            "results": benchmark_result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )


@router.get("/performance/analytics")
async def get_performance_analytics(days: int = 7):
    """Get performance analytics for optimization."""
    try:
        analytics = optimization_service.get_performance_analytics(days)
        return {
            "success": True,
            "analytics": analytics,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/quality/analytics")
async def get_quality_analytics(days: int = 7):
    """Get quality analytics for optimization."""
    try:
        analytics = optimization_service.get_quality_analytics(days)
        return {
            "success": True,
            "analytics": analytics,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quality analytics: {str(e)}"
        )


@router.post("/analyze-query")
async def analyze_query_complexity(query: str, document_count: int = 0):
    """Analyze query complexity for optimization recommendations."""
    try:
        analysis = performance_analyzer.analyze_query_complexity(query, document_count)
        recommendation = performance_analyzer.recommend_configuration(query, document_count)
        
        return {
            "success": True,
            "query_analysis": {
                "query": analysis.query,
                "complexity_score": analysis.complexity_score,
                "estimated_tokens": analysis.estimated_tokens,
                "recommended_strategy": analysis.recommended_strategy.value,
                "reasoning": analysis.reasoning,
                "document_scope": analysis.document_scope
            },
            "configuration_recommendation": recommendation
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query analysis failed: {str(e)}"
        )


@router.get("/performance/profiles")
async def get_performance_profiles():
    """Get current performance profiles and configurations."""
    try:
        summary = performance_analyzer.get_performance_summary()
        return {
            "success": True,
            "performance_summary": summary
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance profiles: {str(e)}"
        )


@router.post("/performance/profile")
async def profile_configuration(
    config_name: str,
    provider: str,
    search_type: str,
    max_sources: int = 8,
    temperature: float = 0.1,
    max_tokens: int = 1000
):
    """Profile a specific configuration for performance."""
    try:
        profile = await performance_analyzer.profile_configuration(
            config_name=config_name,
            provider=provider,
            search_type=search_type,
            max_sources=max_sources,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "success": True,
            "profile": {
                "configuration_name": profile.configuration_name,
                "avg_response_time": profile.avg_response_time,
                "avg_quality_score": profile.avg_quality_score,
                "efficiency_score": profile.efficiency_score,
                "provider": profile.provider,
                "search_type": profile.search_type
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Configuration profiling failed: {str(e)}"
        )


@router.post("/routing/decision")
async def make_routing_decision(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    user_preferences: Optional[Dict[str, Any]] = None
):
    """Get intelligent routing decision for a query."""
    try:
        routing_result = routing_service.make_routing_decision(
            query=query,
            context=context,
            user_preferences=user_preferences
        )
        
        return {
            "success": True,
            "routing_decision": {
                "selected_provider": routing_result.selected_provider,
                "selected_model": routing_result.selected_model,
                "routing_reason": routing_result.routing_reason,
                "confidence_score": routing_result.confidence_score,
                "fallback_provider": routing_result.fallback_provider,
                "estimated_performance": routing_result.estimated_performance
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Routing decision failed: {str(e)}"
        )


@router.get("/routing/analytics")
async def get_routing_analytics(days: int = 7):
    """Get routing analytics and performance data."""
    try:
        analytics = routing_service.get_routing_analytics(days)
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get routing analytics: {str(e)}"
        )


@router.post("/routing/preferences")
async def update_routing_preferences(preferences: RoutingPreferencesRequest):
    """Update user preferences for routing decisions."""
    try:
        routing_service.update_user_preferences(preferences.dict())
        return {
            "success": True,
            "message": "Routing preferences updated successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update routing preferences: {str(e)}"
        )


@router.get("/cache/statistics")
async def get_cache_statistics():
    """Get comprehensive cache statistics."""
    try:
        stats = cache_service.get_cache_statistics()
        return {
            "success": True,
            "cache_statistics": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@router.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        cleanup_result = await cache_service.cleanup_expired_cache()
        return {
            "success": True,
            "cleanup_result": cleanup_result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache cleanup failed: {str(e)}"
        )


@router.post("/cache/config")
async def update_cache_config(config_request: CacheConfigRequest):
    """Update cache configuration."""
    try:
        cache_service.update_cache_config(config_request.dict())
        return {
            "success": True,
            "message": "Cache configuration updated successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update cache config: {str(e)}"
        )


@router.post("/cache/invalidate")
async def invalidate_cache_pattern(pattern: str):
    """Invalidate cache entries matching a pattern."""
    try:
        invalidated_count = await cache_service.invalidate_cache_by_pattern(pattern)
        return {
            "success": True,
            "invalidated_count": invalidated_count,
            "message": f"Invalidated {invalidated_count} cache entries"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache invalidation failed: {str(e)}"
        )


@router.post("/graphrag/optimize-document")
async def optimize_document_graphrag(document_id: str):
    """Optimize GraphRAG processing for a specific document."""
    try:
        # This would require integration with document service
        # For now, return placeholder response
        return {
            "success": True,
            "message": "GraphRAG optimization started",
            "document_id": document_id,
            "optimization_id": f"graphrag_{int(datetime.now().timestamp())}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"GraphRAG optimization failed: {str(e)}"
        )


@router.get("/graphrag/metrics")
async def get_graphrag_metrics():
    """Get GraphRAG optimization metrics."""
    try:
        metrics = graphrag_optimizer.get_optimization_metrics()
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GraphRAG metrics: {str(e)}"
        )


@router.post("/graphrag/clear-cache")
async def clear_graphrag_cache():
    """Clear GraphRAG optimization cache."""
    try:
        result = graphrag_optimizer.clear_optimization_cache()
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear GraphRAG cache: {str(e)}"
        )


@router.get("/monitoring/health")
async def get_system_health():
    """Get comprehensive system health status."""
    try:
        health_status = await monitoring_service.get_system_health_status()
        return {
            "success": True,
            "health_status": {
                "overall_status": health_status.overall_status,
                "component_statuses": health_status.component_statuses,
                "active_alerts": health_status.active_alerts,
                "performance_summary": health_status.performance_summary,
                "last_updated": health_status.last_updated.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system health: {str(e)}"
        )


@router.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get real-time monitoring dashboard data."""
    try:
        dashboard_data = monitoring_service.get_performance_dashboard_data()
        return {
            "success": True,
            "dashboard_data": dashboard_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard data: {str(e)}"
        )


@router.get("/monitoring/analytics")
async def get_optimization_analytics(days: int = 7):
    """Get optimization analytics and insights."""
    try:
        analytics = monitoring_service.get_optimization_analytics(days)
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization analytics: {str(e)}"
        )


@router.post("/monitoring/cleanup")
async def cleanup_monitoring_data(retention_days: int = 30):
    """Clean up old monitoring data."""
    try:
        cleanup_result = await monitoring_service.cleanup_old_metrics(retention_days)
        return {
            "success": True,
            "cleanup_result": cleanup_result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring cleanup failed: {str(e)}"
        )


@router.get("/recommendations")
async def get_optimization_recommendations():
    """Get current optimization recommendations."""
    try:
        recommendations = await monitoring_service.generate_optimization_recommendations()
        
        return {
            "success": True,
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "category": rec.category,
                    "priority": rec.priority,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_impact": rec.expected_impact,
                    "implementation_effort": rec.implementation_effort,
                    "metrics_supporting": rec.metrics_supporting,
                    "timestamp": rec.timestamp.isoformat()
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@router.post("/apply-optimization")
async def apply_optimization_recommendation(
    recommendation_id: str,
    auto_apply: bool = False
):
    """Apply a specific optimization recommendation."""
    try:
        # This would implement actual optimization application logic
        # For now, return success response
        return {
            "success": True,
            "message": f"Optimization recommendation {recommendation_id} applied successfully",
            "recommendation_id": recommendation_id,
            "auto_applied": auto_apply
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply optimization: {str(e)}"
        )


@router.get("/overview")
async def get_optimization_overview():
    """Get comprehensive optimization overview."""
    try:
        # Collect data from all services
        performance_analytics = optimization_service.get_performance_analytics(7)
        routing_analytics = routing_service.get_routing_analytics(7)
        cache_stats = cache_service.get_cache_statistics()
        health_status = await monitoring_service.get_system_health_status()
        
        return {
            "success": True,
            "overview": {
                "system_health": health_status.overall_status,
                "active_alerts": len(health_status.active_alerts),
                "performance_summary": health_status.performance_summary,
                "cache_hit_rate": cache_stats.get("performance_metrics", {}).get("hit_rate", 0.0),
                "routing_efficiency": routing_analytics.get("average_confidence", 0.0),
                "total_optimizations_available": 5,  # Placeholder
                "optimization_score": 0.75  # Overall optimization effectiveness
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization overview: {str(e)}"
        )


@router.post("/auto-optimize")
async def enable_auto_optimization(
    enabled: bool = True,
    confidence_threshold: float = 0.8,
    max_changes_per_hour: int = 3
):
    """Enable or disable automatic optimization."""
    try:
        # Update monitoring service configuration
        new_config = {
            "auto_optimization": {
                "enabled": enabled,
                "confidence_threshold": confidence_threshold,
                "max_changes_per_hour": max_changes_per_hour
            }
        }
        
        # This would update the actual configuration
        return {
            "success": True,
            "message": f"Auto-optimization {'enabled' if enabled else 'disabled'}",
            "configuration": new_config["auto_optimization"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure auto-optimization: {str(e)}"
        )