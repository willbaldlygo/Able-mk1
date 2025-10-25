"""
Staged Reasoning Service for Enhanced Response Generation
Implements multi-stage approach: outline → retrieval → synthesis
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from .ai_service import AIService
from .vector_service import VectorService
from .hybrid_search_service import HybridSearchService

@dataclass
class ReasoningStage:
    stage: str
    content: str
    sources: List[Dict[str, Any]]
    confidence: float

class StagedReasoningService:
    def __init__(self):
        self.ai_service = AIService()
        self.vector_service = VectorService()
        self.hybrid_search = HybridSearchService()
    
    async def generate_staged_response(self, question: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate response using staged reasoning approach"""
        
        # Stage 1: Generate outline and identify key aspects
        outline_stage = await self._generate_outline(question)
        
        # Stage 2: Multi-pass retrieval based on outline
        retrieval_stage = await self._multi_pass_retrieval(question, outline_stage, document_ids)
        
        # Stage 3: Synthesize final response with source diversification
        synthesis_stage = await self._synthesize_response(question, outline_stage, retrieval_stage)
        
        return {
            "answer": synthesis_stage.content,
            "sources": synthesis_stage.sources,
            "reasoning_stages": [outline_stage, retrieval_stage, synthesis_stage],
            "confidence": synthesis_stage.confidence
        }
    
    async def _generate_outline(self, question: str) -> ReasoningStage:
        """Stage 1: Generate response outline and identify key aspects"""
        
        outline_prompt = f"""
        Analyze this question and create a structured outline for a comprehensive response:
        
        Question: {question}
        
        Provide:
        1. Key aspects to address (3-5 main points)
        2. Information types needed (facts, examples, analysis, etc.)
        3. Potential perspectives to consider
        4. Response structure outline
        
        Format as JSON:
        {{
            "key_aspects": ["aspect1", "aspect2", ...],
            "information_types": ["facts", "examples", ...],
            "perspectives": ["perspective1", "perspective2", ...],
            "structure": ["intro", "main_points", "conclusion"]
        }}
        """
        
        outline_response = self.ai_service.generate_response_with_provider(outline_prompt, [])
        
        try:
            outline_data = json.loads(outline_response.answer)
        except:
            # Fallback if JSON parsing fails
            outline_data = {
                "key_aspects": [question],
                "information_types": ["facts"],
                "perspectives": ["general"],
                "structure": ["response"]
            }
        
        return ReasoningStage(
            stage="outline",
            content=json.dumps(outline_data, indent=2),
            sources=[],
            confidence=0.8
        )
    
    async def _multi_pass_retrieval(self, question: str, outline_stage: ReasoningStage, document_ids: Optional[List[str]]) -> ReasoningStage:
        """Stage 2: Multi-pass retrieval for comprehensive information gathering"""
        
        outline_data = json.loads(outline_stage.content)
        all_sources = []
        
        # Pass 1: Overview retrieval
        overview_sources = self.vector_service.strategic_search(
            query=f"Overview of: {question}",
            document_ids=document_ids
        )
        all_sources.extend(overview_sources[:4])  # Limit after retrieval
        
        # Pass 2: Specific aspect retrieval
        for aspect in outline_data.get("key_aspects", []):
            aspect_sources = self.vector_service.strategic_search(
                query=f"{aspect} related to {question}",
                document_ids=document_ids
            )
            all_sources.extend(aspect_sources[:3])  # Limit after retrieval
        
        # Pass 3: Perspective-based retrieval
        for perspective in outline_data.get("perspectives", []):
            perspective_sources = self.vector_service.strategic_search(
                query=f"{question} from {perspective} perspective",
                document_ids=document_ids
            )
            all_sources.extend(perspective_sources[:2])  # Limit after retrieval
        
        # Diversify sources by document
        diversified_sources = self._diversify_sources(all_sources)
        
        return ReasoningStage(
            stage="retrieval",
            content=f"Retrieved {len(diversified_sources)} sources across {len(set(s.document_id for s in diversified_sources))} documents",
            sources=diversified_sources,
            confidence=0.9
        )
    
    async def _synthesize_response(self, question: str, outline_stage: ReasoningStage, retrieval_stage: ReasoningStage) -> ReasoningStage:
        """Stage 3: Synthesize final response using outline and retrieved sources"""
        
        outline_data = json.loads(outline_stage.content)
        sources = retrieval_stage.sources
        
        # Create context from sources
        context_parts = []
        for i, source in enumerate(sources[:12]):  # Limit to top 12 sources
            context_parts.append(f"Source {i+1} ({source.document_name}): {source.chunk_content}")
        
        context = "\n\n".join(context_parts)
        
        synthesis_prompt = f"""
        Using the structured outline and retrieved sources, provide a comprehensive response.
        
        Question: {question}
        
        Outline Structure:
        {outline_stage.content}
        
        Retrieved Information:
        {context}
        
        Instructions:
        1. Follow the outlined structure
        2. Address each key aspect identified
        3. Draw from multiple sources when possible
        4. Provide specific examples and evidence
        5. Consider different perspectives mentioned
        6. Ensure response is well-organized and comprehensive
        
        Provide a detailed, well-structured response that synthesizes information from the sources.
        """
        
        final_response = self.ai_service.generate_response_with_provider(synthesis_prompt, sources)
        
        return ReasoningStage(
            stage="synthesis",
            content=final_response.answer,
            sources=sources,
            confidence=min(0.95, final_response.confidence if hasattr(final_response, 'confidence') else 0.85)
        )
    
    def _diversify_sources(self, sources: List[Any], max_per_document: int = 3) -> List[Any]:
        """Ensure response draws from multiple documents"""
        
        # Group sources by document
        doc_groups = {}
        for source in sources:
            doc_id = source.document_id
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(source)
        
        # Select top sources from each document
        diversified = []
        for doc_id, doc_sources in doc_groups.items():
            # Sort by relevance score and take top N
            sorted_sources = sorted(doc_sources, key=lambda x: x.relevance_score, reverse=True)
            diversified.extend(sorted_sources[:max_per_document])
        
        # Sort final list by relevance
        return sorted(diversified, key=lambda x: x.relevance_score, reverse=True)