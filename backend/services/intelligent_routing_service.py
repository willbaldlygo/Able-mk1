"""Intelligent Query Routing Service for Local vs Cloud AI Selection."""
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from config import config
from models import ChatRequest, EnhancedChatRequest
from services.performance_analyzer import QueryComplexityAnalysis, OptimizationStrategy
from services.ai_service import AIProvider


class RoutingDecision(Enum):
    """Routing decision types."""
    LOCAL_PREFERRED = "local_preferred"
    CLOUD_PREFERRED = "cloud_preferred"
    LOCAL_ONLY = "local_only"
    CLOUD_ONLY = "cloud_only"
    ADAPTIVE = "adaptive"


class SystemLoadLevel(Enum):
    """System resource load levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """Current system performance metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Optional[float]
    disk_io_percent: float
    load_level: SystemLoadLevel
    available_providers: List[str]
    timestamp: datetime


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    query_complexity: QueryComplexityAnalysis
    system_metrics: SystemMetrics
    user_preferences: Dict[str, Any]
    performance_history: Dict[str, List[float]]
    cost_constraints: Optional[Dict[str, float]]
    latency_requirements: Optional[float]


@dataclass
class RoutingResult:
    """Result of routing decision."""
    selected_provider: str
    selected_model: Optional[str]
    routing_reason: str
    confidence_score: float
    fallback_provider: Optional[str]
    estimated_performance: Dict[str, float]
    routing_metadata: Dict[str, Any]


class IntelligentRoutingService:
    """Service for intelligent routing between local and cloud AI providers."""
    
    def __init__(self):
        self.routing_data_dir = config.project_root / "data" / "routing"
        self.routing_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking files
        self.performance_history_file = self.routing_data_dir / "performance_history.json"
        self.routing_decisions_file = self.routing_data_dir / "routing_decisions.json"
        self.user_preferences_file = self.routing_data_dir / "user_preferences.json"
        
        # Load historical data
        self.performance_history = self._load_performance_history()
        self.routing_decisions_history = []
        self.user_preferences = self._load_user_preferences()
        
        # Routing thresholds and weights
        self.routing_config = self._initialize_routing_config()
        
        # Performance analyzer integration (optional)
        try:
            from services.performance_analyzer import PerformanceAnalyzer
            self.performance_analyzer = PerformanceAnalyzer()
        except:
            self.performance_analyzer = None
    
    def _initialize_routing_config(self) -> Dict[str, Any]:
        """Initialize routing configuration with default thresholds."""
        return {
            # System resource thresholds
            "cpu_thresholds": {
                "low": 30.0,
                "medium": 60.0,
                "high": 85.0
            },
            "memory_thresholds": {
                "low": 40.0,
                "medium": 70.0,
                "high": 90.0
            },
            
            # Query complexity thresholds
            "complexity_thresholds": {
                "simple": 0.3,
                "moderate": 0.6,
                "complex": 0.8
            },
            
            # Performance requirements
            "latency_thresholds": {
                "fast": 2.0,      # seconds
                "normal": 5.0,
                "slow": 15.0
            },
            
            # Provider capabilities
            "provider_strengths": {
                "anthropic": {
                    "reasoning": 0.95,
                    "creativity": 0.90,
                    "accuracy": 0.92,
                    "speed": 0.75,
                    "cost_efficiency": 0.60
                },
                "ollama": {
                    "reasoning": 0.75,
                    "creativity": 0.70,
                    "accuracy": 0.80,
                    "speed": 0.85,
                    "cost_efficiency": 1.0  # Local = free
                }
            },
            
            # Routing weights
            "decision_weights": {
                "performance": 0.30,
                "latency": 0.25,
                "cost": 0.20,
                "quality": 0.15,
                "availability": 0.10
            }
        }
    
    def _load_performance_history(self) -> Dict[str, List[Dict]]:
        """Load historical performance data."""
        try:
            if self.performance_history_file.exists():
                with open(self.performance_history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading performance history: {e}")
        
        return {"anthropic": [], "ollama": []}
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences for routing."""
        try:
            if self.user_preferences_file.exists():
                with open(self.user_preferences_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading user preferences: {e}")
        
        return {
            "preferred_provider": "auto",  # auto, anthropic, ollama
            "max_latency": 10.0,          # seconds
            "cost_sensitivity": 0.5,      # 0-1, higher = more cost sensitive
            "quality_priority": 0.8,      # 0-1, higher = prefer quality over speed
            "privacy_preference": 0.3     # 0-1, higher = prefer local processing
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O (simplified)
            disk_io = psutil.disk_io_counters()
            disk_io_percent = min((disk_io.read_bytes + disk_io.write_bytes) / (1024**3), 100) if disk_io else 0
            
            # GPU memory (if available)
            gpu_memory_percent = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory_percent = sum(gpu.memoryUtil for gpu in gpus) / len(gpus) * 100
            except:
                pass
            
            # Determine load level
            load_level = self._determine_load_level(cpu_percent, memory_percent)
            
            # Check available providers
            available_providers = []
            if config.anthropic_api_key:
                available_providers.append("anthropic")
            if config.ollama_enabled:
                available_providers.append("ollama")
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_io_percent=disk_io_percent,
                load_level=load_level,
                available_providers=available_providers,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error getting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=50.0,
                memory_percent=50.0,
                gpu_memory_percent=None,
                disk_io_percent=10.0,
                load_level=SystemLoadLevel.MEDIUM,
                available_providers=["anthropic"] if config.anthropic_api_key else [],
                timestamp=datetime.now()
            )
    
    def _determine_load_level(self, cpu_percent: float, memory_percent: float) -> SystemLoadLevel:
        """Determine system load level based on resource usage."""
        thresholds = self.routing_config["cpu_thresholds"]
        mem_thresholds = self.routing_config["memory_thresholds"]
        
        # Take the higher of CPU and memory load
        max_load = max(cpu_percent, memory_percent)
        
        if max_load >= thresholds["high"] or memory_percent >= mem_thresholds["high"]:
            return SystemLoadLevel.CRITICAL if max_load >= 95 else SystemLoadLevel.HIGH
        elif max_load >= thresholds["medium"]:
            return SystemLoadLevel.MEDIUM
        else:
            return SystemLoadLevel.LOW
    
    def analyze_query_for_routing(self, query: str, document_count: int = 0) -> QueryComplexityAnalysis:
        """Analyze query complexity for routing decisions."""
        if self.performance_analyzer:
            return self.performance_analyzer.analyze_query_complexity(query, document_count)
        
        # Fallback simple analysis
        from services.performance_analyzer import QueryComplexityAnalysis, OptimizationStrategy
        
        complexity_score = min(len(query.split()) / 20, 1.0)  # Simple length-based complexity
        
        return QueryComplexityAnalysis(
            query=query,
            complexity_score=complexity_score,
            estimated_tokens=len(query.split()) * 10,
            recommended_strategy=OptimizationStrategy.BALANCED,
            reasoning="Simple fallback analysis"
        )
    
    def make_routing_decision(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """Make intelligent routing decision based on multiple factors."""
        
        # Get current system state
        system_metrics = self.get_system_metrics()
        query_complexity = self.analyze_query_for_routing(query)
        
        # Merge user preferences
        effective_preferences = {**self.user_preferences}
        if user_preferences:
            effective_preferences.update(user_preferences)
        
        # Create routing context
        routing_context = RoutingContext(
            query_complexity=query_complexity,
            system_metrics=system_metrics,
            user_preferences=effective_preferences,
            performance_history=self.performance_history,
            cost_constraints=context.get('cost_constraints') if context else None,
            latency_requirements=context.get('max_latency') if context else None
        )
        
        # Execute routing decision logic
        routing_result = self._execute_routing_logic(routing_context)
        
        # Log decision for learning
        self._log_routing_decision(query, routing_context, routing_result)
        
        return routing_result
    
    def _execute_routing_logic(self, context: RoutingContext) -> RoutingResult:
        """Execute the core routing decision logic."""
        
        # Check basic availability
        available_providers = context.system_metrics.available_providers
        if not available_providers:
            return RoutingResult(
                selected_provider="none",
                selected_model=None,
                routing_reason="No providers available",
                confidence_score=0.0,
                fallback_provider=None,
                estimated_performance={},
                routing_metadata={}
            )
        
        # Force routing if only one provider available
        if len(available_providers) == 1:
            provider = available_providers[0]
            return RoutingResult(
                selected_provider=provider,
                selected_model=None,
                routing_reason=f"Only {provider} available",
                confidence_score=1.0,
                fallback_provider=None,
                estimated_performance=self._estimate_provider_performance(provider, context),
                routing_metadata={"forced_choice": True}
            )
        
        # Calculate scores for each available provider
        provider_scores = {}
        for provider in available_providers:
            provider_scores[provider] = self._score_provider_for_context(provider, context)
        
        # Select best provider
        best_provider = max(provider_scores, key=provider_scores.get)
        best_score = provider_scores[best_provider]
        
        # Determine fallback
        fallback_provider = None
        if len(provider_scores) > 1:
            remaining_providers = {k: v for k, v in provider_scores.items() if k != best_provider}
            fallback_provider = max(remaining_providers, key=remaining_providers.get)
        
        # Generate reasoning
        routing_reason = self._generate_routing_reasoning(best_provider, context, provider_scores)
        
        return RoutingResult(
            selected_provider=best_provider,
            selected_model=self._select_optimal_model(best_provider, context),
            routing_reason=routing_reason,
            confidence_score=best_score,
            fallback_provider=fallback_provider,
            estimated_performance=self._estimate_provider_performance(best_provider, context),
            routing_metadata={
                "all_scores": provider_scores,
                "system_load": context.system_metrics.load_level.value,
                "query_complexity": context.query_complexity.complexity_score
            }
        )
    
    def _score_provider_for_context(self, provider: str, context: RoutingContext) -> float:
        """Score a provider for the given context."""
        weights = self.routing_config["decision_weights"]
        provider_strengths = self.routing_config["provider_strengths"].get(provider, {})
        
        # Factor 1: Performance capability
        query_complexity = context.query_complexity.complexity_score
        if query_complexity < self.routing_config["complexity_thresholds"]["simple"]:
            # Simple queries - both providers fine, prefer faster/cheaper
            performance_score = provider_strengths.get("speed", 0.5)
        elif query_complexity > self.routing_config["complexity_thresholds"]["complex"]:
            # Complex queries - prefer higher reasoning capability
            performance_score = provider_strengths.get("reasoning", 0.5)
        else:
            # Moderate queries - balance accuracy and speed
            performance_score = (
                provider_strengths.get("accuracy", 0.5) * 0.6 +
                provider_strengths.get("speed", 0.5) * 0.4
            )
        
        # Factor 2: Latency requirements
        latency_score = 1.0
        max_latency = context.latency_requirements or context.user_preferences.get("max_latency", 10.0)
        
        if max_latency <= self.routing_config["latency_thresholds"]["fast"]:
            # Fast response required - prefer local
            latency_score = 1.0 if provider == "ollama" else 0.6
        elif max_latency >= self.routing_config["latency_thresholds"]["slow"]:
            # Slow response acceptable - prefer quality
            latency_score = 1.0 if provider == "anthropic" else 0.8
        
        # Factor 3: Cost considerations
        cost_sensitivity = context.user_preferences.get("cost_sensitivity", 0.5)
        cost_score = 1.0
        if cost_sensitivity > 0.5:
            # Cost sensitive - prefer local
            cost_score = provider_strengths.get("cost_efficiency", 0.5)
        
        # Factor 4: Quality requirements
        quality_priority = context.user_preferences.get("quality_priority", 0.5)
        quality_score = 1.0
        if quality_priority > 0.7:
            # High quality required
            quality_score = (
                provider_strengths.get("accuracy", 0.5) * 0.5 +
                provider_strengths.get("reasoning", 0.5) * 0.5
            )
        
        # Factor 5: System availability and load
        availability_score = 1.0
        if provider == "ollama" and context.system_metrics.load_level in [SystemLoadLevel.HIGH, SystemLoadLevel.CRITICAL]:
            # System under high load - local model may struggle
            availability_score = 0.3
        elif provider == "anthropic" and context.system_metrics.load_level == SystemLoadLevel.LOW:
            # System not busy - cloud API may be overkill for simple tasks
            availability_score = 0.9
        
        # Historical performance adjustment
        historical_score = self._get_historical_performance_score(provider)
        
        # Calculate weighted final score
        final_score = (
            performance_score * weights["performance"] +
            latency_score * weights["latency"] +
            cost_score * weights["cost"] +
            quality_score * weights["quality"] +
            availability_score * weights["availability"]
        )
        
        # Apply historical performance modifier
        final_score *= historical_score
        
        return min(final_score, 1.0)
    
    def _get_historical_performance_score(self, provider: str) -> float:
        """Get performance score modifier based on historical data."""
        if provider not in self.performance_history:
            return 1.0
        
        history = self.performance_history[provider]
        if not history:
            return 1.0
        
        # Recent performance data (last 10 requests)
        recent_history = history[-10:]
        
        # Calculate success rate and average performance
        success_count = sum(1 for entry in recent_history if entry.get('success', True))
        success_rate = success_count / len(recent_history)
        
        # Calculate average response time performance
        response_times = [entry.get('response_time', 5.0) for entry in recent_history]
        avg_response_time = sum(response_times) / len(response_times)
        
        # Score based on success rate and response time
        time_score = max(0.1, min(1.0, 10.0 / avg_response_time))  # Better scores for faster responses
        
        return (success_rate * 0.7) + (time_score * 0.3)
    
    def _select_optimal_model(self, provider: str, context: RoutingContext) -> Optional[str]:
        """Select optimal model for the chosen provider."""
        if provider == "anthropic":
            # For Anthropic, could choose between different Claude models
            complexity = context.query_complexity.complexity_score
            if complexity > 0.8:
                return "claude-3-5-sonnet-20241022"  # Best model for complex queries
            else:
                return config.claude_model  # Default model
        
        elif provider == "ollama":
            # For Ollama, could choose based on system resources and complexity
            load_level = context.system_metrics.load_level
            if load_level in [SystemLoadLevel.HIGH, SystemLoadLevel.CRITICAL]:
                return "llama3.1:8b"  # Smaller model for resource constraints
            else:
                return config.ollama_model  # Default model
        
        return None
    
    def _generate_routing_reasoning(
        self, 
        selected_provider: str, 
        context: RoutingContext, 
        all_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasons = []
        
        # System load reasoning
        if context.system_metrics.load_level == SystemLoadLevel.HIGH and selected_provider == "anthropic":
            reasons.append("high system load favors cloud processing")
        elif context.system_metrics.load_level == SystemLoadLevel.LOW and selected_provider == "ollama":
            reasons.append("low system load allows efficient local processing")
        
        # Query complexity reasoning
        complexity = context.query_complexity.complexity_score
        if complexity > 0.7 and selected_provider == "anthropic":
            reasons.append("complex query benefits from advanced reasoning capabilities")
        elif complexity < 0.3 and selected_provider == "ollama":
            reasons.append("simple query can be handled efficiently locally")
        
        # Cost reasoning
        if context.user_preferences.get("cost_sensitivity", 0) > 0.7 and selected_provider == "ollama":
            reasons.append("cost optimization prefers local processing")
        
        # Latency reasoning
        max_latency = context.latency_requirements or context.user_preferences.get("max_latency", 10)
        if max_latency < 3 and selected_provider == "ollama":
            reasons.append("low latency requirement favors local model")
        
        # Score-based reasoning
        score_diff = max(all_scores.values()) - min(all_scores.values())
        if score_diff < 0.1:
            reasons.append("close performance scores, slight edge to selected provider")
        else:
            reasons.append(f"clear performance advantage (score: {all_scores[selected_provider]:.2f})")
        
        return "; ".join(reasons) if reasons else f"optimal choice based on current conditions"
    
    def _estimate_provider_performance(self, provider: str, context: RoutingContext) -> Dict[str, float]:
        """Estimate performance metrics for the selected provider."""
        provider_strengths = self.routing_config["provider_strengths"].get(provider, {})
        
        # Base estimates
        base_estimates = {
            "response_time": 3.0 if provider == "anthropic" else 2.0,
            "accuracy": provider_strengths.get("accuracy", 0.8),
            "cost_per_query": 0.01 if provider == "anthropic" else 0.0,
        }
        
        # Adjust based on query complexity
        complexity_factor = 1 + context.query_complexity.complexity_score
        base_estimates["response_time"] *= complexity_factor
        
        # Adjust based on system load (for local models)
        if provider == "ollama":
            load_multiplier = {
                SystemLoadLevel.LOW: 1.0,
                SystemLoadLevel.MEDIUM: 1.3,
                SystemLoadLevel.HIGH: 1.8,
                SystemLoadLevel.CRITICAL: 2.5
            }.get(context.system_metrics.load_level, 1.0)
            base_estimates["response_time"] *= load_multiplier
        
        return base_estimates
    
    def _log_routing_decision(
        self, 
        query: str, 
        context: RoutingContext, 
        result: RoutingResult
    ):
        """Log routing decision for future learning."""
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_hash": hash(query) % 10000,  # Hash for privacy
            "query_length": len(query.split()),
            "complexity_score": context.query_complexity.complexity_score,
            "system_load": context.system_metrics.load_level.value,
            "selected_provider": result.selected_provider,
            "confidence_score": result.confidence_score,
            "routing_reason": result.routing_reason
        }
        
        self.routing_decisions_history.append(decision_entry)
        
        # Keep only recent decisions (last 1000)
        if len(self.routing_decisions_history) > 1000:
            self.routing_decisions_history = self.routing_decisions_history[-1000:]
    
    def update_performance_feedback(
        self, 
        provider: str, 
        actual_response_time: float,
        success: bool,
        quality_score: Optional[float] = None
    ):
        """Update performance history with actual results."""
        if provider not in self.performance_history:
            self.performance_history[provider] = []
        
        performance_entry = {
            "timestamp": datetime.now().isoformat(),
            "response_time": actual_response_time,
            "success": success,
            "quality_score": quality_score
        }
        
        self.performance_history[provider].append(performance_entry)
        
        # Keep only recent history (last 100 per provider)
        if len(self.performance_history[provider]) > 100:
            self.performance_history[provider] = self.performance_history[provider][-100:]
        
        # Periodically save to disk
        if len(self.performance_history[provider]) % 10 == 0:
            self._save_performance_history()
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        try:
            with open(self.performance_history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            print(f"Error saving performance history: {e}")
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences for routing."""
        self.user_preferences.update(preferences)
        
        try:
            with open(self.user_preferences_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving user preferences: {e}")
    
    def get_routing_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics on routing decisions."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_decisions = [
            d for d in self.routing_decisions_history
            if datetime.fromisoformat(d["timestamp"]) > cutoff_date
        ]
        
        if not recent_decisions:
            return {"message": "No recent routing decisions"}
        
        # Provider distribution
        provider_counts = {}
        for decision in recent_decisions:
            provider = decision["selected_provider"]
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        # Average confidence
        avg_confidence = sum(d["confidence_score"] for d in recent_decisions) / len(recent_decisions)
        
        # System load distribution
        load_distribution = {}
        for decision in recent_decisions:
            load = decision["system_load"]
            load_distribution[load] = load_distribution.get(load, 0) + 1
        
        return {
            "period_days": days,
            "total_decisions": len(recent_decisions),
            "provider_distribution": provider_counts,
            "average_confidence": avg_confidence,
            "system_load_distribution": load_distribution,
            "performance_feedback": {
                provider: len(history) for provider, history in self.performance_history.items()
            }
        }
    
    def optimize_routing_configuration(self) -> Dict[str, Any]:
        """Optimize routing configuration based on historical performance."""
        optimization_results = {
            "optimizations_applied": [],
            "recommendations": []
        }
        
        # Analyze provider performance over time
        for provider in self.performance_history:
            history = self.performance_history[provider]
            if len(history) < 10:
                continue
            
            recent_history = history[-20:]
            success_rate = sum(1 for h in recent_history if h.get("success", True)) / len(recent_history)
            avg_response_time = sum(h.get("response_time", 5.0) for h in recent_history) / len(recent_history)
            
            # Update provider strength estimates
            current_strengths = self.routing_config["provider_strengths"].get(provider, {})
            
            # Adjust speed rating based on actual performance
            if avg_response_time < 2.0:
                current_strengths["speed"] = min(current_strengths.get("speed", 0.5) + 0.1, 1.0)
            elif avg_response_time > 8.0:
                current_strengths["speed"] = max(current_strengths.get("speed", 0.5) - 0.1, 0.1)
            
            # Adjust accuracy based on success rate
            if success_rate > 0.95:
                current_strengths["accuracy"] = min(current_strengths.get("accuracy", 0.5) + 0.05, 1.0)
            elif success_rate < 0.85:
                current_strengths["accuracy"] = max(current_strengths.get("accuracy", 0.5) - 0.05, 0.1)
            
            self.routing_config["provider_strengths"][provider] = current_strengths
            optimization_results["optimizations_applied"].append(
                f"Updated {provider} performance estimates based on {len(recent_history)} recent requests"
            )
        
        # Generate recommendations based on usage patterns
        recent_decisions = self.routing_decisions_history[-100:] if self.routing_decisions_history else []
        if recent_decisions:
            provider_usage = {}
            for decision in recent_decisions:
                provider = decision["selected_provider"]
                provider_usage[provider] = provider_usage.get(provider, 0) + 1
            
            total_decisions = len(recent_decisions)
            for provider, count in provider_usage.items():
                usage_ratio = count / total_decisions
                if usage_ratio > 0.8:
                    optimization_results["recommendations"].append(
                        f"Consider optimizing {provider} further as it handles {usage_ratio:.1%} of queries"
                    )
                elif usage_ratio < 0.2:
                    optimization_results["recommendations"].append(
                        f"Review {provider} configuration - only handling {usage_ratio:.1%} of queries"
                    )
        
        return optimization_results