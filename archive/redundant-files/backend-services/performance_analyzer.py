"""Performance Analysis Service for Speed vs Accuracy optimization."""
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from config import config
from models import ChatRequest, EnhancedChatRequest, SourceInfo, EnhancedSourceInfo
from services.ai_optimization_service import PerformanceMetrics, ResponseQualityMetrics


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    SPEED_PRIORITY = "speed_priority"      # Minimize response time
    ACCURACY_PRIORITY = "accuracy_priority"  # Maximize response quality
    BALANCED = "balanced"                  # Balance speed and accuracy
    ADAPTIVE = "adaptive"                  # Adapt based on query complexity


@dataclass
class SpeedAccuracyProfile:
    """Speed vs accuracy profile for different configurations."""
    configuration_name: str
    avg_response_time: float
    avg_quality_score: float
    provider: str
    search_type: str
    max_sources: int
    chunk_size: int
    temperature: float
    max_tokens: int
    efficiency_score: float  # Combined speed/accuracy metric


@dataclass
class QueryComplexityAnalysis:
    """Analysis of query complexity to inform optimization decisions."""
    query: str
    complexity_score: float  # 0-1, higher = more complex
    estimated_tokens: int
    recommended_strategy: OptimizationStrategy
    reasoning: str
    entity_count: int = 0
    relationship_count: int = 0
    document_scope: str = "single"  # single, multiple, all


class PerformanceAnalyzer:
    """Analyzer for speed vs accuracy optimization."""
    
    def __init__(self):
        self.optimization_service = None  # Will be injected to avoid circular imports
        self.analysis_cache_dir = config.project_root / "data" / "performance_analysis"
        self.analysis_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance profiles storage
        self.profiles_file = self.analysis_cache_dir / "speed_accuracy_profiles.json"
        self.cached_profiles: List[SpeedAccuracyProfile] = []
        
        # Load existing profiles
        self._load_cached_profiles()
        
        # Performance thresholds
        self.speed_threshold_fast = 2.0  # seconds
        self.speed_threshold_slow = 8.0  # seconds
        self.quality_threshold_high = 0.8  # quality score
        self.quality_threshold_low = 0.6   # quality score
    
    def _load_cached_profiles(self):
        """Load cached performance profiles."""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    data = json.load(f)
                    self.cached_profiles = [
                        SpeedAccuracyProfile(**profile) for profile in data
                    ]
        except Exception as e:
            print(f"Error loading cached profiles: {e}")
            self.cached_profiles = []
    
    def _save_profiles(self):
        """Save performance profiles to cache."""
        try:
            data = [asdict(profile) for profile in self.cached_profiles]
            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving profiles: {e}")
    
    def analyze_query_complexity(self, query: str, document_count: int = 0) -> QueryComplexityAnalysis:
        """Analyze query complexity to determine optimal strategy."""
        
        # Basic complexity indicators
        word_count = len(query.split())
        char_count = len(query)
        question_marks = query.count('?')
        
        # Advanced complexity indicators
        complex_words = [
            'analyze', 'compare', 'contrast', 'synthesize', 'relationship',
            'comprehensive', 'detailed', 'elaborate', 'implications',
            'methodology', 'framework', 'systematic', 'correlation'
        ]
        
        simple_words = [
            'what', 'who', 'when', 'where', 'define', 'list', 'name',
            'basic', 'simple', 'quick', 'brief', 'summary'
        ]
        
        complex_count = sum(1 for word in complex_words if word in query.lower())
        simple_count = sum(1 for word in simple_words if word in query.lower())
        
        # Calculate complexity score
        complexity_factors = {
            'length': min(word_count / 20, 1.0) * 0.3,  # Longer queries more complex
            'complexity_words': min(complex_count / 3, 1.0) * 0.3,
            'document_scope': min(document_count / 10, 1.0) * 0.2,
            'question_complexity': min(question_marks / 3, 1.0) * 0.1,
            'simplicity_penalty': -min(simple_count / 2, 0.5) * 0.1
        }
        
        complexity_score = max(0.1, sum(complexity_factors.values()))
        
        # Estimate token requirements
        estimated_tokens = max(50, word_count * 10 + complex_count * 50)
        
        # Determine document scope
        scope_indicators = {
            'single': ['this document', 'the document', 'in this', 'specific'],
            'multiple': ['documents', 'compare', 'across', 'between'],
            'all': ['all documents', 'overall', 'comprehensive', 'entire collection']
        }
        
        document_scope = "single"
        for scope, indicators in scope_indicators.items():
            if any(indicator in query.lower() for indicator in indicators):
                document_scope = scope
                break
        
        # Recommend strategy
        if complexity_score < 0.3:
            strategy = OptimizationStrategy.SPEED_PRIORITY
            reasoning = "Simple query - prioritize speed"
        elif complexity_score > 0.7:
            strategy = OptimizationStrategy.ACCURACY_PRIORITY
            reasoning = "Complex query - prioritize accuracy"
        elif document_count > 5:
            strategy = OptimizationStrategy.BALANCED
            reasoning = "Multiple documents - balance speed and accuracy"
        else:
            strategy = OptimizationStrategy.ADAPTIVE
            reasoning = "Moderate complexity - use adaptive approach"
        
        return QueryComplexityAnalysis(
            query=query,
            complexity_score=complexity_score,
            estimated_tokens=estimated_tokens,
            recommended_strategy=strategy,
            reasoning=reasoning,
            document_scope=document_scope
        )
    
    async def profile_configuration(
        self,
        config_name: str,
        provider: str,
        search_type: str,
        max_sources: int = 8,
        chunk_size: int = 600,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        test_queries: Optional[List[str]] = None
    ) -> SpeedAccuracyProfile:
        """Profile a specific configuration for speed vs accuracy."""
        
        if test_queries is None:
            test_queries = self._get_standard_test_queries()
        
        response_times = []
        quality_scores = []
        
        print(f"Profiling configuration: {config_name}")
        
        for i, query in enumerate(test_queries):
            print(f"  Testing query {i+1}/{len(test_queries)}...")
            
            start_time = time.time()
            
            try:
                # Create test request
                if search_type in ['global', 'local']:
                    request = EnhancedChatRequest(
                        question=query,
                        search_type=search_type
                    )
                    # Would execute enhanced search here
                    response_time = time.time() - start_time
                    
                    # Mock quality score for now (would use actual response evaluation)
                    quality_score = 0.7 + (search_type == 'global') * 0.1
                    
                else:  # vector search
                    request = ChatRequest(question=query)
                    # Would execute vector search here
                    response_time = time.time() - start_time
                    
                    # Mock quality score
                    quality_score = 0.65
                
                # Adjust for configuration parameters
                response_time *= (max_sources / 8)  # More sources = more time
                response_time *= (chunk_size / 600)  # Larger chunks = more time
                response_time *= (max_tokens / 1000)  # More tokens = more time
                
                quality_score += (max_sources - 5) * 0.02  # More sources = better quality
                quality_score += (max_tokens - 500) / 1000 * 0.1  # More tokens = better quality
                quality_score = min(quality_score, 1.0)
                
                response_times.append(response_time)
                quality_scores.append(quality_score)
                
            except Exception as e:
                print(f"    Error testing query: {e}")
                # Use penalty values for failed queries
                response_times.append(10.0)
                quality_scores.append(0.3)
        
        # Calculate averages
        avg_response_time = statistics.mean(response_times)
        avg_quality_score = statistics.mean(quality_scores)
        
        # Calculate efficiency score (balance of speed and quality)
        # Higher is better - normalize response time (lower is better) and quality (higher is better)
        normalized_speed = max(0, 1 - (avg_response_time / 15))  # 15s = worst case
        efficiency_score = (normalized_speed + avg_quality_score) / 2
        
        profile = SpeedAccuracyProfile(
            configuration_name=config_name,
            avg_response_time=avg_response_time,
            avg_quality_score=avg_quality_score,
            provider=provider,
            search_type=search_type,
            max_sources=max_sources,
            chunk_size=chunk_size,
            temperature=temperature,
            max_tokens=max_tokens,
            efficiency_score=efficiency_score
        )
        
        # Cache the profile
        self.cached_profiles.append(profile)
        self._save_profiles()
        
        return profile
    
    def _get_standard_test_queries(self) -> List[str]:
        """Get standard test queries for profiling."""
        return [
            "What is the main topic?",  # Simple
            "Summarize the key findings across all documents.",  # Medium
            "How do the methodologies compare between different studies?",  # Complex
            "Who are the main authors mentioned?",  # Simple
            "What are the implications of these findings for future research?",  # Complex
        ]
    
    async def benchmark_all_configurations(self) -> Dict[str, SpeedAccuracyProfile]:
        """Benchmark all reasonable configuration combinations."""
        configurations = []
        
        # Define configuration space
        providers = ["anthropic"]
        if config.ollama_enabled:
            providers.append("ollama")
        
        search_types = ["vector", "local", "global"]
        max_sources_options = [3, 5, 8, 12]
        temperature_options = [0.0, 0.1, 0.3]
        max_tokens_options = [500, 1000, 1500]
        
        # Generate configurations (limit to avoid excessive testing)
        for provider in providers:
            for search_type in search_types:
                for max_sources in max_sources_options:
                    for temp in temperature_options:
                        for tokens in max_tokens_options:
                            config_name = f"{provider}_{search_type}_src{max_sources}_temp{temp}_tok{tokens}"
                            configurations.append({
                                'name': config_name,
                                'provider': provider,
                                'search_type': search_type,
                                'max_sources': max_sources,
                                'temperature': temp,
                                'max_tokens': tokens
                            })
        
        # Limit configurations for practical testing
        configurations = configurations[:20]  # Test top 20 configurations
        
        results = {}
        for i, config_def in enumerate(configurations):
            print(f"Benchmarking configuration {i+1}/{len(configurations)}: {config_def['name']}")
            
            profile = await self.profile_configuration(
                config_name=config_def['name'],
                provider=config_def['provider'],
                search_type=config_def['search_type'],
                max_sources=config_def['max_sources'],
                temperature=config_def['temperature'],
                max_tokens=config_def['max_tokens']
            )
            
            results[config_def['name']] = profile
        
        return results
    
    def get_optimal_configuration(
        self, 
        strategy: OptimizationStrategy,
        query_complexity: Optional[QueryComplexityAnalysis] = None
    ) -> Optional[SpeedAccuracyProfile]:
        """Get the optimal configuration for a given strategy."""
        
        if not self.cached_profiles:
            return None
        
        if strategy == OptimizationStrategy.SPEED_PRIORITY:
            # Find fastest configuration above minimum quality
            valid_profiles = [
                p for p in self.cached_profiles 
                if p.avg_quality_score >= self.quality_threshold_low
            ]
            if valid_profiles:
                return min(valid_profiles, key=lambda p: p.avg_response_time)
        
        elif strategy == OptimizationStrategy.ACCURACY_PRIORITY:
            # Find highest quality configuration below maximum time
            valid_profiles = [
                p for p in self.cached_profiles 
                if p.avg_response_time <= self.speed_threshold_slow
            ]
            if valid_profiles:
                return max(valid_profiles, key=lambda p: p.avg_quality_score)
        
        elif strategy == OptimizationStrategy.BALANCED:
            # Find best efficiency score
            return max(self.cached_profiles, key=lambda p: p.efficiency_score)
        
        elif strategy == OptimizationStrategy.ADAPTIVE:
            # Choose based on query complexity
            if query_complexity:
                if query_complexity.complexity_score < 0.4:
                    return self.get_optimal_configuration(OptimizationStrategy.SPEED_PRIORITY)
                elif query_complexity.complexity_score > 0.7:
                    return self.get_optimal_configuration(OptimizationStrategy.ACCURACY_PRIORITY)
                else:
                    return self.get_optimal_configuration(OptimizationStrategy.BALANCED)
        
        # Fallback to balanced
        return self.get_optimal_configuration(OptimizationStrategy.BALANCED)
    
    def recommend_configuration(self, query: str, document_count: int = 0) -> Dict[str, Any]:
        """Recommend optimal configuration for a specific query."""
        
        # Analyze query complexity
        complexity_analysis = self.analyze_query_complexity(query, document_count)
        
        # Get optimal configuration
        optimal_config = self.get_optimal_configuration(
            complexity_analysis.recommended_strategy,
            complexity_analysis
        )
        
        if not optimal_config:
            return {
                'error': 'No performance profiles available. Run benchmark first.',
                'complexity_analysis': asdict(complexity_analysis)
            }
        
        return {
            'query': query,
            'complexity_analysis': asdict(complexity_analysis),
            'recommended_configuration': asdict(optimal_config),
            'performance_prediction': {
                'estimated_response_time': optimal_config.avg_response_time,
                'estimated_quality_score': optimal_config.avg_quality_score,
                'confidence': 'medium'  # Could be calculated based on profile variance
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance profiles."""
        if not self.cached_profiles:
            return {'message': 'No performance profiles available'}
        
        # Group by strategy effectiveness
        speed_profiles = sorted(self.cached_profiles, key=lambda p: p.avg_response_time)
        quality_profiles = sorted(self.cached_profiles, key=lambda p: p.avg_quality_score, reverse=True)
        balanced_profiles = sorted(self.cached_profiles, key=lambda p: p.efficiency_score, reverse=True)
        
        return {
            'total_profiles': len(self.cached_profiles),
            'fastest_configuration': asdict(speed_profiles[0]) if speed_profiles else None,
            'highest_quality_configuration': asdict(quality_profiles[0]) if quality_profiles else None,
            'most_balanced_configuration': asdict(balanced_profiles[0]) if balanced_profiles else None,
            'average_response_time': statistics.mean([p.avg_response_time for p in self.cached_profiles]),
            'average_quality_score': statistics.mean([p.avg_quality_score for p in self.cached_profiles]),
            'configuration_range': {
                'response_time_range': [
                    min(p.avg_response_time for p in self.cached_profiles),
                    max(p.avg_response_time for p in self.cached_profiles)
                ],
                'quality_range': [
                    min(p.avg_quality_score for p in self.cached_profiles),
                    max(p.avg_quality_score for p in self.cached_profiles)
                ]
            }
        }
    
    async def continuous_optimization(self, query: str, document_count: int = 0) -> Dict[str, Any]:
        """Continuously optimize configuration based on real-time performance."""
        
        # Get current recommendation
        recommendation = self.recommend_configuration(query, document_count)
        
        # If we have recent performance data, adjust recommendation
        recent_profiles = [
            p for p in self.cached_profiles
            if hasattr(p, 'timestamp') and 
            p.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if recent_profiles:
            # Adjust based on recent performance trends
            recent_avg_time = statistics.mean([p.avg_response_time for p in recent_profiles])
            recent_avg_quality = statistics.mean([p.avg_quality_score for p in recent_profiles])
            
            # If recent performance is degrading, suggest more conservative settings
            if recent_avg_time > self.speed_threshold_slow:
                recommendation['optimization_suggestion'] = "Consider reducing max_sources and max_tokens for better response times"
            elif recent_avg_quality < self.quality_threshold_low:
                recommendation['optimization_suggestion'] = "Consider increasing max_sources and search depth for better quality"
            else:
                recommendation['optimization_suggestion'] = "Current configuration performing well"
            
            recommendation['recent_performance'] = {
                'avg_response_time': recent_avg_time,
                'avg_quality_score': recent_avg_quality,
                'sample_size': len(recent_profiles)
            }
        
        return recommendation
    
    def export_performance_report(self) -> Dict[str, Any]:
        """Export comprehensive performance analysis report."""
        return {
            'report_generated': datetime.now().isoformat(),
            'performance_summary': self.get_performance_summary(),
            'all_profiles': [asdict(p) for p in self.cached_profiles],
            'optimization_recommendations': self._generate_global_recommendations(),
            'configuration_comparison': self._compare_configurations()
        }
    
    def _generate_global_recommendations(self) -> List[str]:
        """Generate global optimization recommendations."""
        recommendations = []
        
        if not self.cached_profiles:
            return ["Run performance benchmarks to get optimization recommendations"]
        
        # Analyze patterns across profiles
        avg_response_time = statistics.mean([p.avg_response_time for p in self.cached_profiles])
        avg_quality = statistics.mean([p.avg_quality_score for p in self.cached_profiles])
        
        if avg_response_time > self.speed_threshold_slow:
            recommendations.append("Overall system performance is slow - consider reducing default max_sources and max_tokens")
        
        if avg_quality < self.quality_threshold_high:
            recommendations.append("Overall response quality could be improved - consider better prompt engineering or higher-quality models")
        
        # Provider-specific recommendations
        provider_performance = {}
        for profile in self.cached_profiles:
            if profile.provider not in provider_performance:
                provider_performance[profile.provider] = []
            provider_performance[profile.provider].append(profile.efficiency_score)
        
        for provider, scores in provider_performance.items():
            avg_score = statistics.mean(scores)
            if avg_score > 0.7:
                recommendations.append(f"{provider} provider shows strong performance - consider as primary option")
            elif avg_score < 0.5:
                recommendations.append(f"{provider} provider shows weak performance - consider optimization or alternative")
        
        return recommendations
    
    def _compare_configurations(self) -> Dict[str, Any]:
        """Compare different configuration aspects."""
        if not self.cached_profiles:
            return {}
        
        # Group by different dimensions
        by_provider = {}
        by_search_type = {}
        by_source_count = {}
        
        for profile in self.cached_profiles:
            # By provider
            if profile.provider not in by_provider:
                by_provider[profile.provider] = []
            by_provider[profile.provider].append(profile.efficiency_score)
            
            # By search type
            if profile.search_type not in by_search_type:
                by_search_type[profile.search_type] = []
            by_search_type[profile.search_type].append(profile.efficiency_score)
            
            # By source count
            if profile.max_sources not in by_source_count:
                by_source_count[profile.max_sources] = []
            by_source_count[profile.max_sources].append(profile.efficiency_score)
        
        return {
            'by_provider': {k: statistics.mean(v) for k, v in by_provider.items()},
            'by_search_type': {k: statistics.mean(v) for k, v in by_search_type.items()},
            'by_source_count': {str(k): statistics.mean(v) for k, v in by_source_count.items()}
        }