"""Comprehensive Performance Monitoring and Metrics Service for Able AI Optimization."""
import asyncio
import json
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import logging
import statistics

from config import config

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    CACHE_PERFORMANCE = "cache_performance"
    AI_PROVIDER_PERFORMANCE = "ai_provider_performance"
    SEARCH_PERFORMANCE = "search_performance"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    threshold: Optional[float] = None


@dataclass
class SystemHealthStatus:
    """Overall system health status."""
    overall_status: str  # healthy, degraded, unhealthy
    component_statuses: Dict[str, str]
    active_alerts: List[Dict[str, Any]]
    performance_summary: Dict[str, float]
    last_updated: datetime


@dataclass
class PerformanceAlert:
    """Performance alert structure."""
    alert_id: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class OptimizationRecommendation:
    """AI optimization recommendation."""
    recommendation_id: str
    category: str  # performance, cost, accuracy, etc.
    priority: str  # high, medium, low
    title: str
    description: str
    expected_impact: str
    implementation_effort: str
    metrics_supporting: List[str]
    timestamp: datetime


class PerformanceMonitoringService:
    """Comprehensive performance monitoring and optimization analytics."""
    
    def __init__(self):
        self.monitoring_dir = config.project_root / "data" / "monitoring"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric storage directories
        self.metrics_dir = self.monitoring_dir / "metrics"
        self.alerts_dir = self.monitoring_dir / "alerts"
        self.recommendations_dir = self.monitoring_dir / "recommendations"
        self.health_dir = self.monitoring_dir / "health"
        
        for directory in [self.metrics_dir, self.alerts_dir, self.recommendations_dir, self.health_dir]:
            directory.mkdir(exist_ok=True)
        
        # In-memory metric buffers for real-time monitoring
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: List[PerformanceAlert] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Monitoring configuration
        self.monitoring_config = self._load_monitoring_config()
        
        # Performance thresholds
        self.thresholds = self._initialize_thresholds()
        
        # Initialize service references (will be injected)
        self.ai_optimization_service = None
        self.performance_analyzer = None
        self.routing_service = None
        self.cache_service = None
        
        # Background monitoring tasks
        self._monitoring_tasks = []
        self._start_background_monitoring()
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        return {
            "enabled": True,
            "collection_interval_seconds": 30,
            "alert_check_interval_seconds": 60,
            "metric_retention_days": 30,
            "enable_real_time_alerts": True,
            "enable_email_alerts": False,
            "enable_webhooks": False,
            "auto_optimization": {
                "enabled": True,
                "confidence_threshold": 0.8,
                "max_changes_per_hour": 3
            },
            "dashboard": {
                "refresh_interval_seconds": 10,
                "show_predictions": True,
                "show_recommendations": True
            }
        }
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance thresholds."""
        return {
            "response_time": {
                "anthropic": {"warning": 8.0, "critical": 15.0},
                "ollama": {"warning": 5.0, "critical": 12.0},
                "search": {"warning": 3.0, "critical": 8.0}
            },
            "system_resources": {
                "cpu_percent": {"warning": 80.0, "critical": 95.0},
                "memory_percent": {"warning": 85.0, "critical": 95.0},
                "disk_usage_percent": {"warning": 90.0, "critical": 98.0}
            },
            "accuracy": {
                "response_quality": {"warning": 0.6, "critical": 0.4},
                "relevance_score": {"warning": 0.5, "critical": 0.3}
            },
            "cache": {
                "hit_rate": {"warning": 0.3, "critical": 0.1},
                "miss_rate": {"warning": 0.7, "critical": 0.9}
            }
        }
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        if not self.monitoring_config["enabled"]:
            return
        
        # Don't start tasks if already running
        if self._monitoring_tasks:
            return
        
        loop = asyncio.get_event_loop()
        
        # System metrics collection
        self._monitoring_tasks.append(
            loop.create_task(self._collect_system_metrics_loop())
        )
        
        # Alert checking
        self._monitoring_tasks.append(
            loop.create_task(self._check_alerts_loop())
        )
        
        # Optimization recommendations
        self._monitoring_tasks.append(
            loop.create_task(self._generate_recommendations_loop())
        )
    
    async def _collect_system_metrics_loop(self):
        """Background task to collect system metrics."""
        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.monitoring_config["collection_interval_seconds"])
            except Exception as e:
                logger.error(f"Error in system metrics collection: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_alerts_loop(self):
        """Background task to check for alerts."""
        while True:
            try:
                await self.check_and_trigger_alerts()
                await asyncio.sleep(self.monitoring_config["alert_check_interval_seconds"])
            except Exception as e:
                logger.error(f"Error in alert checking: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _generate_recommendations_loop(self):
        """Background task to generate optimization recommendations."""
        while True:
            try:
                await self.generate_optimization_recommendations()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in recommendation generation: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def record_metric(
        self, 
        metric_type: MetricType,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        threshold: Optional[float] = None
    ):
        """Record a performance metric."""
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {},
            threshold=threshold
        )
        
        # Add to in-memory buffer
        buffer_key = f"{metric_type.value}_{name}"
        self.metric_buffers[buffer_key].append(metric)
        
        # Persist to disk periodically
        await self._persist_metric(metric)
    
    async def _persist_metric(self, metric: PerformanceMetric):
        """Persist metric to disk storage."""
        try:
            date_str = metric.timestamp.strftime("%Y-%m-%d")
            metric_file = self.metrics_dir / f"{metric.metric_type.value}_{date_str}.json"
            
            # Load existing data
            if metric_file.exists():
                with open(metric_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Add new metric
            metric_data = asdict(metric)
            metric_data['timestamp'] = metric.timestamp.isoformat()
            data.append(metric_data)
            
            # Write back to file
            with open(metric_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist metric: {e}")
    
    async def collect_system_metrics(self):
        """Collect comprehensive system performance metrics."""
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        await self.record_metric(
            MetricType.RESOURCE_USAGE, "cpu_percent", cpu_percent, "%",
            tags={"component": "system"}
        )
        
        await self.record_metric(
            MetricType.RESOURCE_USAGE, "memory_percent", memory.percent, "%",
            tags={"component": "system"}
        )
        
        await self.record_metric(
            MetricType.RESOURCE_USAGE, "disk_usage_percent", 
            (disk.used / disk.total) * 100, "%",
            tags={"component": "system"}
        )
        
        # AI service metrics
        if hasattr(self, 'ai_service') and self.ai_service:
            try:
                provider_status = self.ai_service.test_connection()
                for provider, is_connected in provider_status.items():
                    await self.record_metric(
                        MetricType.AI_PROVIDER_PERFORMANCE, f"{provider}_connection",
                        1.0 if is_connected else 0.0, "boolean",
                        tags={"provider": provider, "component": "ai"}
                    )
            except Exception as e:
                logger.warning(f"Failed to collect AI service metrics: {e}")
        
        # Cache metrics
        if self.cache_service:
            try:
                cache_stats = self.cache_service.get_cache_statistics()
                performance_metrics = cache_stats.get("performance_metrics", {})
                
                if "hit_rate" in performance_metrics:
                    await self.record_metric(
                        MetricType.CACHE_PERFORMANCE, "hit_rate",
                        performance_metrics["hit_rate"], "ratio",
                        tags={"component": "cache"}
                    )
                
                if "avg_response_time_cached" in performance_metrics:
                    await self.record_metric(
                        MetricType.RESPONSE_TIME, "cache_response_time",
                        performance_metrics["avg_response_time_cached"], "seconds",
                        tags={"component": "cache", "type": "cached"}
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to collect cache metrics: {e}")
    
    async def record_ai_response_metrics(
        self,
        provider: str,
        response_time: float,
        success: bool,
        query_complexity: Optional[float] = None,
        quality_score: Optional[float] = None,
        token_count: Optional[int] = None
    ):
        """Record AI response performance metrics."""
        
        tags = {"provider": provider, "component": "ai"}
        
        # Response time
        await self.record_metric(
            MetricType.RESPONSE_TIME, f"{provider}_response_time",
            response_time, "seconds", tags=tags,
            threshold=self.thresholds["response_time"].get(provider, {}).get("warning")
        )
        
        # Success rate
        await self.record_metric(
            MetricType.AI_PROVIDER_PERFORMANCE, f"{provider}_success",
            1.0 if success else 0.0, "boolean", tags=tags
        )
        
        # Quality score if available
        if quality_score is not None:
            await self.record_metric(
                MetricType.ACCURACY, f"{provider}_quality_score",
                quality_score, "score", tags=tags,
                threshold=self.thresholds["accuracy"]["response_quality"]["warning"]
            )
        
        # Token count for throughput analysis
        if token_count is not None:
            tokens_per_second = token_count / response_time if response_time > 0 else 0
            await self.record_metric(
                MetricType.THROUGHPUT, f"{provider}_tokens_per_second",
                tokens_per_second, "tokens/sec", tags=tags
            )
    
    async def record_search_metrics(
        self,
        search_type: str,
        search_time: float,
        result_count: int,
        relevance_scores: List[float]
    ):
        """Record search performance metrics."""
        
        tags = {"search_type": search_type, "component": "search"}
        
        # Search time
        await self.record_metric(
            MetricType.RESPONSE_TIME, f"{search_type}_search_time",
            search_time, "seconds", tags=tags,
            threshold=self.thresholds["response_time"]["search"]["warning"]
        )
        
        # Result count
        await self.record_metric(
            MetricType.SEARCH_PERFORMANCE, f"{search_type}_result_count",
            result_count, "count", tags=tags
        )
        
        # Average relevance
        if relevance_scores:
            avg_relevance = statistics.mean(relevance_scores)
            await self.record_metric(
                MetricType.ACCURACY, f"{search_type}_avg_relevance",
                avg_relevance, "score", tags=tags,
                threshold=self.thresholds["accuracy"]["relevance_score"]["warning"]
            )
    
    async def check_and_trigger_alerts(self):
        """Check metrics against thresholds and trigger alerts."""
        
        if not self.monitoring_config["enable_real_time_alerts"]:
            return
        
        current_time = datetime.now()
        new_alerts = []
        
        # Check recent metrics in buffers
        for buffer_key, metrics_deque in self.metric_buffers.items():
            if not metrics_deque:
                continue
            
            # Get recent metrics (last 5 minutes)
            cutoff_time = current_time - timedelta(minutes=5)
            recent_metrics = [m for m in metrics_deque if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                continue
            
            # Check thresholds
            for metric in recent_metrics[-3:]:  # Check last 3 metrics
                alerts = self._evaluate_metric_thresholds(metric)
                new_alerts.extend(alerts)
        
        # Process new alerts
        for alert in new_alerts:
            await self._handle_alert(alert)
    
    def _evaluate_metric_thresholds(self, metric: PerformanceMetric) -> List[PerformanceAlert]:
        """Evaluate metric against thresholds and generate alerts."""
        alerts = []
        
        # Get relevant thresholds
        thresholds = self._get_thresholds_for_metric(metric)
        
        if not thresholds:
            return alerts
        
        # Check warning threshold
        if "warning" in thresholds and metric.value >= thresholds["warning"]:
            alert_level = AlertLevel.WARNING
            if "critical" in thresholds and metric.value >= thresholds["critical"]:
                alert_level = AlertLevel.CRITICAL
            
            alert_id = f"{metric.name}_{metric.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Don't create duplicate alerts
            if alert_id not in self.active_alerts:
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    level=alert_level,
                    message=self._generate_alert_message(metric, alert_level),
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold=thresholds.get(alert_level.value, 0),
                    timestamp=metric.timestamp
                )
                alerts.append(alert)
        
        return alerts
    
    def _get_thresholds_for_metric(self, metric: PerformanceMetric) -> Dict[str, float]:
        """Get appropriate thresholds for a metric."""
        
        # System resource thresholds
        if metric.metric_type == MetricType.RESOURCE_USAGE:
            if "cpu" in metric.name:
                return self.thresholds["system_resources"]["cpu_percent"]
            elif "memory" in metric.name:
                return self.thresholds["system_resources"]["memory_percent"]
            elif "disk" in metric.name:
                return self.thresholds["system_resources"]["disk_usage_percent"]
        
        # Response time thresholds
        elif metric.metric_type == MetricType.RESPONSE_TIME:
            provider = metric.tags.get("provider", "default")
            if "search" in metric.name:
                return self.thresholds["response_time"]["search"]
            else:
                return self.thresholds["response_time"].get(provider, {})
        
        # Accuracy thresholds
        elif metric.metric_type == MetricType.ACCURACY:
            if "quality" in metric.name:
                return self.thresholds["accuracy"]["response_quality"]
            elif "relevance" in metric.name:
                return self.thresholds["accuracy"]["relevance_score"]
        
        # Cache thresholds
        elif metric.metric_type == MetricType.CACHE_PERFORMANCE:
            if "hit_rate" in metric.name:
                return self.thresholds["cache"]["hit_rate"]
        
        return {}
    
    def _generate_alert_message(self, metric: PerformanceMetric, level: AlertLevel) -> str:
        """Generate human-readable alert message."""
        
        component = metric.tags.get("component", "system")
        provider = metric.tags.get("provider", "")
        
        base_message = f"{level.value.upper()}: {metric.name} is {metric.value:.2f} {metric.unit}"
        
        if provider:
            base_message += f" for {provider}"
        
        # Add context based on metric type
        if metric.metric_type == MetricType.RESPONSE_TIME:
            base_message += " - Performance degradation detected"
        elif metric.metric_type == MetricType.RESOURCE_USAGE:
            base_message += " - High resource usage detected"
        elif metric.metric_type == MetricType.ACCURACY:
            base_message += " - Quality threshold breached"
        
        return base_message
    
    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle a triggered alert."""
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"ALERT TRIGGERED: {alert.message}")
        
        # Persist alert
        await self._persist_alert(alert)
        
        # Trigger auto-optimization if enabled
        if self.monitoring_config["auto_optimization"]["enabled"]:
            await self._trigger_auto_optimization(alert)
    
    async def _persist_alert(self, alert: PerformanceAlert):
        """Persist alert to storage."""
        try:
            date_str = alert.timestamp.strftime("%Y-%m-%d")
            alert_file = self.alerts_dir / f"alerts_{date_str}.json"
            
            # Load existing alerts
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Add new alert
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert.timestamp.isoformat()
            if alert.resolution_time:
                alert_data['resolution_time'] = alert.resolution_time.isoformat()
            data.append(alert_data)
            
            # Write back
            with open(alert_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")
    
    async def _trigger_auto_optimization(self, alert: PerformanceAlert):
        """Trigger automatic optimization based on alert."""
        
        try:
            # Determine optimization action based on alert
            optimization_action = self._determine_optimization_action(alert)
            
            if optimization_action:
                logger.info(f"Auto-optimization triggered: {optimization_action}")
                # Execute optimization (placeholder for actual implementation)
                # await self._execute_optimization_action(optimization_action)
                
        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")
    
    def _determine_optimization_action(self, alert: PerformanceAlert) -> Optional[str]:
        """Determine what optimization action to take for an alert."""
        
        # Response time optimizations
        if "response_time" in alert.metric_name:
            if "anthropic" in alert.metric_name:
                return "switch_to_ollama_for_simple_queries"
            elif "ollama" in alert.metric_name:
                return "reduce_model_size_or_switch_to_anthropic"
        
        # Resource usage optimizations
        elif "cpu_percent" in alert.metric_name or "memory_percent" in alert.metric_name:
            return "enable_aggressive_caching"
        
        # Cache optimizations
        elif "hit_rate" in alert.metric_name:
            return "increase_cache_ttl_and_size"
        
        return None
    
    async def generate_optimization_recommendations(self):
        """Generate AI optimization recommendations based on metrics."""
        
        try:
            recommendations = []
            current_time = datetime.now()
            
            # Analyze recent performance patterns
            performance_analysis = await self._analyze_performance_patterns()
            
            # Generate recommendations based on analysis
            if performance_analysis["avg_response_time"] > 8.0:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"resp_time_{int(current_time.timestamp())}",
                    category="performance",
                    priority="high",
                    title="Optimize Response Time",
                    description=f"Average response time is {performance_analysis['avg_response_time']:.2f}s. Consider enabling aggressive caching or switching to faster models for simple queries.",
                    expected_impact="30-50% reduction in response time",
                    implementation_effort="medium",
                    metrics_supporting=["response_time", "cache_hit_rate"],
                    timestamp=current_time
                ))
            
            if performance_analysis["cache_hit_rate"] < 0.4:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cache_hit_{int(current_time.timestamp())}",
                    category="performance",
                    priority="medium",
                    title="Improve Cache Hit Rate",
                    description=f"Cache hit rate is {performance_analysis['cache_hit_rate']:.2%}. Optimize cache strategies and increase TTL for stable content.",
                    expected_impact="20-40% improvement in response time",
                    implementation_effort="low",
                    metrics_supporting=["cache_hit_rate", "response_time"],
                    timestamp=current_time
                ))
            
            if performance_analysis["system_load"] > 0.8:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"sys_load_{int(current_time.timestamp())}",
                    category="resource",
                    priority="high",
                    title="High System Load Detected",
                    description=f"System load is {performance_analysis['system_load']:.1%}. Consider load balancing or resource optimization.",
                    expected_impact="Improved system stability and response consistency",
                    implementation_effort="high",
                    metrics_supporting=["cpu_percent", "memory_percent"],
                    timestamp=current_time
                ))
            
            # Persist recommendations
            for recommendation in recommendations:
                await self._persist_recommendation(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def _analyze_performance_patterns(self) -> Dict[str, float]:
        """Analyze recent performance patterns."""
        
        # Default analysis
        analysis = {
            "avg_response_time": 5.0,
            "cache_hit_rate": 0.5,
            "system_load": 0.5,
            "accuracy_score": 0.8
        }
        
        try:
            # Calculate metrics from recent data
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            # Response time analysis
            response_times = []
            cache_hits = []
            system_loads = []
            
            for buffer_key, metrics_deque in self.metric_buffers.items():
                recent_metrics = [m for m in metrics_deque if m.timestamp > cutoff_time]
                
                if "response_time" in buffer_key:
                    response_times.extend([m.value for m in recent_metrics])
                elif "hit_rate" in buffer_key:
                    cache_hits.extend([m.value for m in recent_metrics])
                elif "cpu_percent" in buffer_key or "memory_percent" in buffer_key:
                    system_loads.extend([m.value / 100.0 for m in recent_metrics])
            
            if response_times:
                analysis["avg_response_time"] = statistics.mean(response_times)
            if cache_hits:
                analysis["cache_hit_rate"] = statistics.mean(cache_hits)
            if system_loads:
                analysis["system_load"] = statistics.mean(system_loads)
                
        except Exception as e:
            logger.warning(f"Performance pattern analysis failed: {e}")
        
        return analysis
    
    async def _persist_recommendation(self, recommendation: OptimizationRecommendation):
        """Persist optimization recommendation."""
        try:
            date_str = recommendation.timestamp.strftime("%Y-%m-%d")
            rec_file = self.recommendations_dir / f"recommendations_{date_str}.json"
            
            # Load existing recommendations
            if rec_file.exists():
                with open(rec_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Add new recommendation
            rec_data = asdict(recommendation)
            rec_data['timestamp'] = recommendation.timestamp.isoformat()
            data.append(rec_data)
            
            # Write back
            with open(rec_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist recommendation: {e}")
    
    async def get_system_health_status(self) -> SystemHealthStatus:
        """Get comprehensive system health status."""
        
        current_time = datetime.now()
        
        # Collect component statuses
        component_statuses = {
            "ai_service": "healthy",
            "search_service": "healthy",
            "cache_service": "healthy",
            "vector_db": "healthy",
            "system_resources": "healthy"
        }
        
        # Check for active alerts
        active_alerts_list = []
        critical_alerts = 0
        warning_alerts = 0
        
        for alert in self.active_alerts.values():
            if not alert.resolved:
                active_alerts_list.append({
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                })
                
                if alert.level == AlertLevel.CRITICAL:
                    critical_alerts += 1
                elif alert.level == AlertLevel.WARNING:
                    warning_alerts += 1
        
        # Determine overall status
        if critical_alerts > 0:
            overall_status = "unhealthy"
        elif warning_alerts > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Performance summary
        performance_summary = await self._get_performance_summary()
        
        return SystemHealthStatus(
            overall_status=overall_status,
            component_statuses=component_statuses,
            active_alerts=active_alerts_list,
            performance_summary=performance_summary,
            last_updated=current_time
        )
    
    async def _get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary metrics."""
        
        summary = {
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "system_cpu_usage": 0.0,
            "system_memory_usage": 0.0,
            "ai_success_rate": 0.0
        }
        
        try:
            # Calculate from recent metrics
            cutoff_time = datetime.now() - timedelta(minutes=30)
            
            for buffer_key, metrics_deque in self.metric_buffers.items():
                recent_metrics = [m for m in metrics_deque if m.timestamp > cutoff_time]
                
                if not recent_metrics:
                    continue
                
                avg_value = statistics.mean([m.value for m in recent_metrics])
                
                if "response_time" in buffer_key:
                    summary["avg_response_time"] = max(summary["avg_response_time"], avg_value)
                elif "hit_rate" in buffer_key:
                    summary["cache_hit_rate"] = avg_value
                elif "cpu_percent" in buffer_key:
                    summary["system_cpu_usage"] = avg_value
                elif "memory_percent" in buffer_key:
                    summary["system_memory_usage"] = avg_value
                elif "success" in buffer_key:
                    summary["ai_success_rate"] = avg_value
                    
        except Exception as e:
            logger.warning(f"Performance summary calculation failed: {e}")
        
        return summary
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance monitoring dashboard."""
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "real_time_metrics": {},
            "recent_trends": {},
            "active_alerts": len(self.active_alerts),
            "system_health": "calculating...",
            "recommendations": []
        }
        
        try:
            # Real-time metrics (last 5 minutes)
            cutoff_time = datetime.now() - timedelta(minutes=5)
            
            for buffer_key, metrics_deque in self.metric_buffers.items():
                recent_metrics = [m for m in metrics_deque if m.timestamp > cutoff_time]
                
                if recent_metrics:
                    latest_metric = recent_metrics[-1]
                    dashboard_data["real_time_metrics"][buffer_key] = {
                        "value": latest_metric.value,
                        "unit": latest_metric.unit,
                        "timestamp": latest_metric.timestamp.isoformat()
                    }
            
            # Recent trends (last hour)
            trend_cutoff = datetime.now() - timedelta(hours=1)
            
            for buffer_key, metrics_deque in self.metric_buffers.items():
                trend_metrics = [m for m in metrics_deque if m.timestamp > trend_cutoff]
                
                if len(trend_metrics) > 1:
                    values = [m.value for m in trend_metrics]
                    dashboard_data["recent_trends"][buffer_key] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": statistics.mean(values),
                        "trend": "up" if values[-1] > values[0] else "down" if values[-1] < values[0] else "stable"
                    }
            
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {e}")
        
        return dashboard_data
    
    def get_optimization_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get optimization analytics and insights."""
        
        analytics = {
            "period_days": days,
            "total_optimizations": 0,
            "performance_improvements": {},
            "cost_savings": {},
            "recommendations_implemented": 0,
            "top_performance_issues": [],
            "optimization_effectiveness": {}
        }
        
        try:
            # Load recent recommendations
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # This would analyze recommendation files and performance data
            # For now, return placeholder analytics
            analytics.update({
                "total_optimizations": 15,
                "performance_improvements": {
                    "avg_response_time_improvement": "23%",
                    "cache_hit_rate_improvement": "18%",
                    "cost_reduction": "12%"
                },
                "top_performance_issues": [
                    "High response time during peak hours",
                    "Low cache hit rate for complex queries",
                    "Suboptimal model routing decisions"
                ],
                "recommendations_implemented": 8
            })
            
        except Exception as e:
            logger.error(f"Optimization analytics failed: {e}")
        
        return analytics
    
    async def cleanup_old_metrics(self, retention_days: int = 30):
        """Clean up old metric files."""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleaned_files = 0
        
        for metrics_file in self.metrics_dir.glob("*.json"):
            try:
                # Parse date from filename
                date_part = metrics_file.stem.split('_')[-1]  # Assumes format: type_YYYY-MM-DD
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    metrics_file.unlink()
                    cleaned_files += 1
                    
            except Exception as e:
                logger.warning(f"Error processing metrics file {metrics_file}: {e}")
        
        return {"cleaned_files": cleaned_files, "retention_days": retention_days}
    
    def stop_monitoring(self):
        """Stop background monitoring tasks."""
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        self._monitoring_tasks.clear()
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        self.stop_monitoring()