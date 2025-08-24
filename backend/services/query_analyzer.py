"""Intelligent query analysis for strategic document retrieval."""
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class QueryIntent:
    """Represents the analyzed intent of a user query."""
    query_type: str  # summary, comparison, methodology, results, definition, etc.
    topics: List[str]  # key topics/concepts
    scope: str  # broad, specific, detailed
    search_strategies: List[str]  # different search approaches to try
    content_preferences: List[str]  # preferred types of content

class QueryAnalyzer:
    """Analyzes user queries to determine optimal search strategy."""
    
    def __init__(self):
        self.query_patterns = {
            'summary': [
                r'summarize?|summary|overview|main points?|key findings?',
                r'what (?:is|are|does)|tell me about',
                r'describe|explain briefly'
            ],
            'comparison': [
                r'compare|comparison|versus|vs\.?|difference|similar',
                r'how (?:do|does) .* differ',
                r'contrast|relate|relationship'
            ],
            'methodology': [
                r'how (?:did|do|does)|method|approach|technique|process',
                r'procedure|steps?|implementation|design',
                r'experiment|study design|framework'
            ],
            'results': [
                r'results?|findings?|outcome|conclusion',
                r'what (?:did|were) .* found?|discovered',
                r'evidence|data|statistics|numbers'
            ],
            'definition': [
                r'what is|define|definition|meaning',
                r'concept of|term|refers? to'
            ],
            'analysis': [
                r'analyz|interpret|understand|insight',
                r'why|because|reason|factor|cause',
                r'implication|significance|impact'
            ]
        }
        
        self.content_indicators = {
            'introduction': ['introduction', 'background', 'overview', 'context'],
            'methodology': ['method', 'approach', 'design', 'procedure', 'framework'],
            'results': ['results', 'findings', 'outcome', 'data', 'analysis'],
            'conclusion': ['conclusion', 'summary', 'implications', 'discussion'],
            'theoretical': ['theory', 'model', 'concept', 'framework', 'principle']
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze a user query to determine search strategy."""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        # Extract topics/keywords
        topics = self._extract_topics(query)
        
        # Determine scope
        scope = self._determine_scope(query_lower)
        
        # Generate search strategies
        search_strategies = self._generate_search_strategies(query_type, topics, query)
        
        # Determine content preferences
        content_preferences = self._determine_content_preferences(query_type)
        
        return QueryIntent(
            query_type=query_type,
            topics=topics,
            scope=scope,
            search_strategies=search_strategies,
            content_preferences=content_preferences
        )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        return 'general'
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract key topics from the query."""
        # Remove common words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
            'their', 'can', 'tell', 'explain', 'describe', 'summarize'
        }
        
        # Extract words, filter stop words, keep meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        topics = [word for word in words if word not in stop_words]
        
        # Also look for phrases in quotes or compound terms
        phrases = re.findall(r'"([^"]*)"', query)
        topics.extend([phrase.lower() for phrase in phrases])
        
        return list(set(topics))
    
    def _determine_scope(self, query: str) -> str:
        """Determine if query asks for broad or specific information."""
        broad_indicators = ['overview', 'summary', 'general', 'broad', 'overall']
        specific_indicators = ['specific', 'particular', 'exact', 'precise', 'detailed']
        
        if any(indicator in query for indicator in broad_indicators):
            return 'broad'
        elif any(indicator in query for indicator in specific_indicators):
            return 'specific'
        else:
            return 'medium'
    
    def _generate_search_strategies(self, query_type: str, topics: List[str], original_query: str) -> List[str]:
        """Generate multiple search strategies based on query analysis."""
        strategies = [original_query]  # Always include original
        
        if query_type == 'summary':
            strategies.extend([
                f"introduction {' '.join(topics[:3])}",
                f"overview {' '.join(topics[:3])}",
                f"abstract {' '.join(topics[:3])}"
            ])
        elif query_type == 'methodology':
            strategies.extend([
                f"methodology {' '.join(topics[:3])}",
                f"methods {' '.join(topics[:3])}",
                f"approach {' '.join(topics[:3])}"
            ])
        elif query_type == 'results':
            strategies.extend([
                f"results {' '.join(topics[:3])}",
                f"findings {' '.join(topics[:3])}",
                f"conclusions {' '.join(topics[:3])}"
            ])
        elif query_type == 'comparison':
            strategies.extend([
                f"comparison {' '.join(topics[:3])}",
                f"differences {' '.join(topics[:3])}",
                f"contrast {' '.join(topics[:3])}"
            ])
        
        # Add topic-focused searches
        for topic in topics[:2]:  # Limit to avoid too many searches
            strategies.append(topic)
        
        return strategies
    
    def _determine_content_preferences(self, query_type: str) -> List[str]:
        """Determine what types of content to prioritize."""
        preferences = {
            'summary': ['introduction', 'conclusion', 'abstract'],
            'methodology': ['methodology', 'methods', 'approach'],
            'results': ['results', 'findings', 'data'],
            'comparison': ['analysis', 'discussion', 'comparison'],
            'definition': ['introduction', 'theoretical', 'background'],
            'analysis': ['discussion', 'analysis', 'implications']
        }
        
        return preferences.get(query_type, ['introduction', 'results', 'conclusion'])