"""AI Optimization Service for Able - Performance Enhancement Framework."""
import json
import time
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from config import config
from models import ChatRequest, EnhancedChatRequest, SourceInfo, EnhancedSourceInfo


class AIProvider(Enum):
    """AI provider types."""
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class SearchStrategy(Enum):
    """Search strategy types."""
    VECTOR = "vector"
    LOCAL = "local"
    GLOBAL = "global"
    AUTO = "auto"


@dataclass
class PerformanceMetrics:
    """Performance metrics for AI responses."""
    response_time: float
    token_count: int
    accuracy_score: Optional[float] = None
    relevance_score: float = 0.0
    source_count: int = 0
    search_type: str = "vector"
    provider: str = "anthropic"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ResponseQualityMetrics:
    """Response quality assessment metrics."""
    completeness_score: float  # How complete is the answer (0-1)
    accuracy_score: float      # How accurate is the information (0-1)
    relevance_score: float     # How relevant to the query (0-1)
    coherence_score: float     # How coherent is the response (0-1)
    source_attribution: float  # How well sources are attributed (0-1)
    confidence_score: float    # Overall confidence (0-1)
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'completeness': 0.20,
            'accuracy': 0.25,
            'relevance': 0.25,
            'coherence': 0.15,
            'source_attribution': 0.10,
            'confidence': 0.05
        }
        
        return (
            self.completeness_score * weights['completeness'] +
            self.accuracy_score * weights['accuracy'] +
            self.relevance_score * weights['relevance'] +
            self.coherence_score * weights['coherence'] +
            self.source_attribution * weights['source_attribution'] +
            self.confidence_score * weights['confidence']
        )


@dataclass
class AIOptimizationResult:
    """Result of AI optimization analysis."""
    provider: str
    search_strategy: str
    performance_metrics: PerformanceMetrics
    quality_metrics: ResponseQualityMetrics
    response_content: str
    sources_used: List[str]
    optimization_suggestions: List[str]


class AIOptimizationService:
    """Service for optimizing AI performance and response quality."""
    
    def __init__(self):
        self.optimization_data_dir = config.project_root / "data" / "optimization"
        self.optimization_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_log_file = self.optimization_data_dir / "performance_metrics.json"
        self.quality_log_file = self.optimization_data_dir / "quality_metrics.json"
        self.benchmark_file = self.optimization_data_dir / "benchmarks.json"
        
        # Cache for recent performance data
        self._performance_cache = []
        self._quality_cache = []
        
        # Load existing data
        self._load_cached_data()
    
    def _load_cached_data(self):
        """Load existing performance and quality data."""
        try:
            if self.performance_log_file.exists():
                with open(self.performance_log_file, 'r') as f:
                    data = json.load(f)
                    self._performance_cache = [
                        PerformanceMetrics(**item) for item in data[-1000:]  # Keep last 1000
                    ]
        except Exception:
            pass
        
        try:
            if self.quality_log_file.exists():
                with open(self.quality_log_file, 'r') as f:
                    data = json.load(f)
                    self._quality_cache = [
                        ResponseQualityMetrics(**item) for item in data[-1000:]  # Keep last 1000
                    ]
        except Exception:
            pass
    
    async def compare_ai_providers(self, request: ChatRequest) -> Dict[str, AIOptimizationResult]:
        """Compare response quality across different AI providers."""
        results = {}
        
        # Test Anthropic Claude (current default)
        if config.anthropic_api_key:
            results["anthropic"] = await self._test_provider_performance(
                request, AIProvider.ANTHROPIC
            )
        
        # Test Ollama if available
        if config.ollama_enabled:
            results["ollama"] = await self._test_provider_performance(
                request, AIProvider.OLLAMA
            )
        
        return results
    
    async def _test_provider_performance(
        self, 
        request: ChatRequest, 
        provider: AIProvider
    ) -> AIOptimizationResult:
        """Test performance of a specific AI provider."""
        start_time = time.time()
        
        try:
            if provider == AIProvider.ANTHROPIC:
                response = await self._get_anthropic_response(request)
            else:  # OLLAMA
                response = await self._get_ollama_response(request)
            
            response_time = time.time() - start_time
            
            # Calculate performance metrics
            performance = PerformanceMetrics(
                response_time=response_time,
                token_count=len(response.answer.split()),
                relevance_score=self._calculate_relevance_score(request.question, response.answer),
                source_count=len(response.sources),
                provider=provider.value
            )
            
            # Calculate quality metrics
            quality = await self._assess_response_quality(
                request.question, response.answer, response.sources
            )
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(performance, quality)
            
            return AIOptimizationResult(
                provider=provider.value,
                search_strategy="vector",  # Default for comparison
                performance_metrics=performance,
                quality_metrics=quality,
                response_content=response.answer,
                sources_used=[src.document_name for src in response.sources],
                optimization_suggestions=suggestions
            )
        
        except Exception as e:
            # Return error result
            return AIOptimizationResult(
                provider=provider.value,
                search_strategy="vector",
                performance_metrics=PerformanceMetrics(
                    response_time=time.time() - start_time,
                    token_count=0,
                    provider=provider.value
                ),
                quality_metrics=ResponseQualityMetrics(
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    relevance_score=0.0,
                    coherence_score=0.0,
                    source_attribution=0.0,
                    confidence_score=0.0
                ),
                response_content=f"Error: {str(e)}",
                sources_used=[],
                optimization_suggestions=[f"Fix provider error: {str(e)}"]
            )
    
    async def _get_anthropic_response(self, request: ChatRequest):
        """Get response from Anthropic Claude."""
        from services.ai_service import AIService
        ai_service = AIService()
        return await ai_service.generate_response(request)
    
    async def _get_ollama_response(self, request: ChatRequest):
        """Get response from Ollama using integrated model service."""
        from services.model_service import model_service
        from models import ChatResponse, SourceInfo
        
        try:
            # Use the integrated Ollama service
            response = model_service.generate_response(
                model_name=model_service.ollama_model,
                prompt=request.question,
                stream=False
            )
            
            if response.get('success', False):
                return ChatResponse(
                    answer=response.get('response', 'No response generated'),
                    sources=[],
                    timestamp=datetime.now()
                )
            else:
                return ChatResponse(
                    answer=f"Ollama error: {response.get('error', 'Unknown error')}",
                    sources=[],
                    timestamp=datetime.now()
                )
        except Exception as e:
            return ChatResponse(
                answer=f"Ollama integration error: {str(e)}",
                sources=[],
                timestamp=datetime.now()
            )
    
    async def compare_search_strategies(
        self, 
        request: EnhancedChatRequest
    ) -> Dict[str, AIOptimizationResult]:
        """Compare response quality across different search strategies."""
        results = {}
        
        # Test each search strategy
        for strategy in [SearchStrategy.VECTOR, SearchStrategy.LOCAL, SearchStrategy.GLOBAL]:
            test_request = EnhancedChatRequest(
                question=request.question,
                document_ids=request.document_ids,
                search_type=strategy.value,
                include_entities=request.include_entities,
                include_relationships=request.include_relationships
            )
            
            results[strategy.value] = await self._test_search_strategy_performance(
                test_request, strategy
            )
        
        return results
    
    async def _test_search_strategy_performance(
        self,
        request: EnhancedChatRequest,
        strategy: SearchStrategy
    ) -> AIOptimizationResult:
        """Test performance of a specific search strategy."""
        start_time = time.time()
        
        try:
            from services.hybrid_search_service import HybridSearchService
            hybrid_service = HybridSearchService()
            
            response = await hybrid_service.intelligent_search(request)
            response_time = time.time() - start_time
            
            # Calculate performance metrics
            performance = PerformanceMetrics(
                response_time=response_time,
                token_count=len(response.answer.split()),
                relevance_score=self._calculate_relevance_score(request.question, response.answer),
                source_count=len(response.sources),
                search_type=strategy.value,
                provider="anthropic"  # Assuming Anthropic for strategy comparison
            )
            
            # Calculate quality metrics
            quality = await self._assess_response_quality(
                request.question, response.answer, response.sources
            )
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(performance, quality)
            
            return AIOptimizationResult(
                provider="anthropic",
                search_strategy=strategy.value,
                performance_metrics=performance,
                quality_metrics=quality,
                response_content=response.answer,
                sources_used=[src.document_name for src in response.sources],
                optimization_suggestions=suggestions
            )
        
        except Exception as e:
            return AIOptimizationResult(
                provider="anthropic",
                search_strategy=strategy.value,
                performance_metrics=PerformanceMetrics(
                    response_time=time.time() - start_time,
                    token_count=0,
                    search_type=strategy.value
                ),
                quality_metrics=ResponseQualityMetrics(
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    relevance_score=0.0,
                    coherence_score=0.0,
                    source_attribution=0.0,
                    confidence_score=0.0
                ),
                response_content=f"Error: {str(e)}",
                sources_used=[],
                optimization_suggestions=[f"Fix strategy error: {str(e)}"]
            )
    
    def _calculate_relevance_score(self, question: str, answer: str) -> float:
        """Calculate relevance score between question and answer."""
        # Simple keyword-based relevance (could be enhanced with embeddings)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(question_words.intersection(answer_words))
        return min(overlap / len(question_words), 1.0)
    
    async def _assess_response_quality(
        self,
        question: str,
        answer: str,
        sources: List[Union[SourceInfo, EnhancedSourceInfo]]
    ) -> ResponseQualityMetrics:
        """Assess the quality of a response."""
        
        # Basic quality assessment (could be enhanced with ML models)
        completeness = self._assess_completeness(question, answer)
        accuracy = self._assess_accuracy(answer, sources)
        relevance = self._calculate_relevance_score(question, answer)
        coherence = self._assess_coherence(answer)
        source_attribution = self._assess_source_attribution(answer, sources)
        confidence = (completeness + accuracy + relevance + coherence) / 4
        
        return ResponseQualityMetrics(
            completeness_score=completeness,
            accuracy_score=accuracy,
            relevance_score=relevance,
            coherence_score=coherence,
            source_attribution=source_attribution,
            confidence_score=confidence
        )
    
    def _assess_completeness(self, question: str, answer: str) -> float:
        """Assess how complete the answer is."""
        # Basic heuristics for completeness
        answer_length = len(answer.split())
        
        # Longer answers tend to be more complete (with diminishing returns)
        if answer_length < 10:
            return 0.3
        elif answer_length < 50:
            return 0.6
        elif answer_length < 150:
            return 0.8
        else:
            return 0.9
    
    def _assess_accuracy(self, answer: str, sources: List[Union[SourceInfo, EnhancedSourceInfo]]) -> float:
        """Assess accuracy based on source alignment."""
        if not sources:
            return 0.5  # Neutral score without sources
        
        # Check if answer content aligns with sources
        source_content = " ".join([src.chunk_content for src in sources])
        answer_words = set(answer.lower().split())
        source_words = set(source_content.lower().split())
        
        if not answer_words:
            return 0.0
        
        # Calculate alignment
        alignment = len(answer_words.intersection(source_words)) / len(answer_words)
        return min(alignment * 1.5, 1.0)  # Boost alignment score
    
    def _assess_coherence(self, answer: str) -> float:
        """Assess coherence of the answer."""
        # Basic coherence assessment
        sentences = answer.split('.')
        
        if len(sentences) < 2:
            return 0.7  # Single sentence is coherent but limited
        
        # Check for transition words and logical flow
        coherence_indicators = [
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'in contrast', 'similarly', 'consequently', 'as a result', 'meanwhile'
        ]
        
        indicator_count = sum(1 for indicator in coherence_indicators 
                            if indicator in answer.lower())
        
        # Normalize based on answer length
        coherence_score = 0.6 + (indicator_count / len(sentences)) * 0.4
        return min(coherence_score, 1.0)
    
    def _assess_source_attribution(self, answer: str, sources: List[Union[SourceInfo, EnhancedSourceInfo]]) -> float:
        """Assess how well sources are attributed in the answer."""
        if not sources:
            return 0.0
        
        # Check for explicit source references
        attribution_indicators = [
            'according to', 'based on', 'from', 'in', 'document', 'source',
            'paper', 'study', 'research', 'report'
        ]
        
        indicator_count = sum(1 for indicator in attribution_indicators 
                            if indicator in answer.lower())
        
        # Also check if document names are mentioned
        doc_mentions = sum(1 for src in sources 
                          if src.document_name.lower() in answer.lower())
        
        attribution_score = (indicator_count + doc_mentions * 2) / (len(sources) + 2)
        return min(attribution_score, 1.0)
    
    def _generate_optimization_suggestions(
        self,
        performance: PerformanceMetrics,
        quality: ResponseQualityMetrics
    ) -> List[str]:
        """Generate optimization suggestions based on metrics."""
        suggestions = []
        
        # Performance-based suggestions
        if performance.response_time > 5.0:
            suggestions.append("Consider caching frequently asked questions")
            suggestions.append("Optimize search result count to reduce processing time")
        
        if performance.source_count < 3:
            suggestions.append("Increase search result count for better source coverage")
        elif performance.source_count > 10:
            suggestions.append("Consider reducing search results to improve processing speed")
        
        # Quality-based suggestions
        if quality.completeness_score < 0.6:
            suggestions.append("Use broader search terms to capture more relevant information")
            suggestions.append("Consider increasing max tokens for more complete responses")
        
        if quality.accuracy_score < 0.7:
            suggestions.append("Improve source filtering to ensure higher quality content")
            suggestions.append("Consider using stricter relevance thresholds")
        
        if quality.coherence_score < 0.7:
            suggestions.append("Adjust temperature settings for more coherent responses")
            suggestions.append("Consider using structured prompts for better organization")
        
        if quality.source_attribution < 0.5:
            suggestions.append("Enhance prompts to encourage better source citation")
            suggestions.append("Consider post-processing to add source references")
        
        return suggestions
    
    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics for analysis."""
        self._performance_cache.append(metrics)
        
        # Keep cache size manageable
        if len(self._performance_cache) > 1000:
            self._performance_cache = self._performance_cache[-1000:]
        
        # Periodic save to disk
        if len(self._performance_cache) % 10 == 0:
            self._save_performance_data()
    
    def log_quality_metrics(self, metrics: ResponseQualityMetrics):
        """Log quality metrics for analysis."""
        self._quality_cache.append(metrics)
        
        # Keep cache size manageable
        if len(self._quality_cache) > 1000:
            self._quality_cache = self._quality_cache[-1000:]
        
        # Periodic save to disk
        if len(self._quality_cache) % 10 == 0:
            self._save_quality_data()
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        try:
            data = [asdict(metrics) for metrics in self._performance_cache]
            # Convert datetime objects to strings for JSON serialization
            for item in data:
                if isinstance(item.get('timestamp'), datetime):
                    item['timestamp'] = item['timestamp'].isoformat()
            
            with open(self.performance_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")
    
    def _save_quality_data(self):
        """Save quality data to disk."""
        try:
            data = [asdict(metrics) for metrics in self._quality_cache]
            with open(self.quality_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving quality data: {e}")
    
    def get_performance_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance analytics for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self._performance_cache 
            if m.timestamp > cutoff_date
        ]
        
        if not recent_metrics:
            return {"message": "No recent performance data"}
        
        response_times = [m.response_time for m in recent_metrics]
        token_counts = [m.token_count for m in recent_metrics]
        relevance_scores = [m.relevance_score for m in recent_metrics]
        
        return {
            "period_days": days,
            "total_queries": len(recent_metrics),
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "avg_token_count": statistics.mean(token_counts),
            "avg_relevance_score": statistics.mean(relevance_scores),
            "provider_breakdown": self._get_provider_breakdown(recent_metrics),
            "search_type_breakdown": self._get_search_type_breakdown(recent_metrics)
        }
    
    def get_quality_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get quality analytics for the last N days."""
        if not self._quality_cache:
            return {"message": "No quality data available"}
        
        recent_metrics = self._quality_cache[-min(len(self._quality_cache), days * 10):]
        
        overall_scores = [m.overall_score for m in recent_metrics]
        completeness_scores = [m.completeness_score for m in recent_metrics]
        accuracy_scores = [m.accuracy_score for m in recent_metrics]
        relevance_scores = [m.relevance_score for m in recent_metrics]
        coherence_scores = [m.coherence_score for m in recent_metrics]
        
        return {
            "total_assessments": len(recent_metrics),
            "avg_overall_score": statistics.mean(overall_scores),
            "avg_completeness": statistics.mean(completeness_scores),
            "avg_accuracy": statistics.mean(accuracy_scores),
            "avg_relevance": statistics.mean(relevance_scores),
            "avg_coherence": statistics.mean(coherence_scores),
            "quality_trend": self._calculate_quality_trend(recent_metrics)
        }
    
    def _get_provider_breakdown(self, metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """Get breakdown of queries by AI provider."""
        breakdown = {}
        for metric in metrics:
            provider = metric.provider
            breakdown[provider] = breakdown.get(provider, 0) + 1
        return breakdown
    
    def _get_search_type_breakdown(self, metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """Get breakdown of queries by search type."""
        breakdown = {}
        for metric in metrics:
            search_type = metric.search_type
            breakdown[search_type] = breakdown.get(search_type, 0) + 1
        return breakdown
    
    def _calculate_quality_trend(self, metrics: List[ResponseQualityMetrics]) -> str:
        """Calculate quality trend over time."""
        if len(metrics) < 10:
            return "insufficient_data"
        
        # Split into two halves and compare
        mid = len(metrics) // 2
        first_half = metrics[:mid]
        second_half = metrics[mid:]
        
        first_avg = statistics.mean([m.overall_score for m in first_half])
        second_avg = statistics.mean([m.overall_score for m in second_half])
        
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    async def run_benchmark_suite(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Run a comprehensive benchmark suite."""
        if test_queries is None:
            test_queries = self._get_default_test_queries()
        
        results = {
            "benchmark_time": datetime.now().isoformat(),
            "test_queries": test_queries,
            "provider_results": {},
            "strategy_results": {},
            "overall_recommendations": []
        }
        
        # Test each query across providers and strategies
        for i, query in enumerate(test_queries):
            print(f"Testing query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            request = ChatRequest(question=query)
            enhanced_request = EnhancedChatRequest(question=query)
            
            # Test providers
            provider_results = await self.compare_ai_providers(request)
            results["provider_results"][f"query_{i}"] = {
                "query": query,
                "results": {k: asdict(v) for k, v in provider_results.items()}
            }
            
            # Test search strategies
            strategy_results = await self.compare_search_strategies(enhanced_request)
            results["strategy_results"][f"query_{i}"] = {
                "query": query,
                "results": {k: asdict(v) for k, v in strategy_results.items()}
            }
        
        # Generate overall recommendations
        results["overall_recommendations"] = self._generate_overall_recommendations(results)
        
        # Save benchmark results
        benchmark_file = self.optimization_data_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _get_default_test_queries(self) -> List[str]:
        """Get default test queries for benchmarking."""
        return [
            "What are the main findings in the uploaded documents?",
            "Summarize the key concepts discussed across all documents.",
            "How do the documents relate to each other?",
            "What specific methodologies are mentioned?",
            "Who are the key people or organizations referenced?",
            "What are the limitations or challenges identified?",
            "What future research directions are suggested?",
            "Can you provide a detailed analysis of the main topic?"
        ]
    
    def _generate_overall_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate overall optimization recommendations from benchmark results."""
        recommendations = []
        
        # Analyze provider performance
        # (This would be more sophisticated with actual benchmark data)
        recommendations.append("Monitor response times and adjust provider based on query complexity")
        recommendations.append("Consider implementing response caching for frequently asked questions")
        recommendations.append("Use GraphRAG local search for entity-specific queries")
        recommendations.append("Use GraphRAG global search for broad research questions")
        recommendations.append("Fall back to vector search for simple factual queries")
        
        return recommendations