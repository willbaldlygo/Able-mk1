import re
import json
from typing import Dict, Optional, Tuple, List
from anthropic import Anthropic
import os

class ResponseFormatter:
    """Detects formatting instructions in user prompts and formats responses accordingly."""
    
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.word_count_pattern = r'(\d+)[-\s]*word'
        
    def detect_format_requirements(self, prompt: str) -> Dict[str, any]:
        """Generate structural template using LLM analysis."""
        try:
            template = self._generate_structure_template(prompt)
            print(f"Generated template: {template}")  # Debug
            return {'template': template, 'use_template': True}
        except Exception as e:
            print(f"Template generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic requirements
            requirements = {}
            word_match = re.search(self.word_count_pattern, prompt.lower())
            if word_match:
                requirements['word_count'] = int(word_match.group(1))
            return requirements
    
    def format_response(self, content: str, requirements: Dict[str, any]) -> str:
        """Format response using LLM-generated template or fallback formatting."""
        if not requirements:
            print("No requirements found, returning original content")
            return content
            
        if requirements.get('use_template') and requirements.get('template'):
            print("Using template formatting")
            return self._format_with_template(content, requirements['template'])
        
        print("Using fallback formatting")
        # Fallback formatting (simplified)
        return content.strip()
    
    def _format_essay(self, content: str, requirements: Dict[str, any]) -> str:
        """Format content as a proper essay with title, introduction, body, and conclusion."""
        # Extract title from content or create one
        title = self._extract_essay_title(content)
        
        # Split content into logical sections
        sections = self._identify_essay_sections(content)
        
        # Build formatted essay
        formatted_parts = []
        
        # Add title
        if title:
            formatted_parts.append(f"# {title}\n")
        
        # Add introduction
        if sections.get('introduction'):
            formatted_parts.append(f"## Introduction\n\n{sections['introduction']}\n")
        
        # Add body paragraphs with subheadings
        if sections.get('body'):
            body_paragraphs = self._split_into_paragraphs(sections['body'])
            for i, paragraph in enumerate(body_paragraphs, 1):
                if len(body_paragraphs) > 2:  # Only add subheadings for longer essays
                    heading = self._generate_paragraph_heading(paragraph, i)
                    formatted_parts.append(f"## {heading}\n\n{paragraph}\n")
                else:
                    formatted_parts.append(f"{paragraph}\n")
        
        # Add conclusion
        if sections.get('conclusion'):
            formatted_parts.append(f"## Conclusion\n\n{sections['conclusion']}")
        
        return '\n'.join(formatted_parts)
    
    def _extract_essay_title(self, content: str) -> str:
        """Extract or generate an appropriate essay title."""
        # Look for existing title patterns
        title_patterns = [
            r'^#\s*(.+?)\n',  # Existing markdown title
            r'^(.+?):\s*(.+?)(?:\n|$)',  # "Topic: Description" format
            r'essay on (.+?)(?:\.|\n)',  # "essay on [topic]"
            r'benefits of (.+?)(?:\s+for|\.|\n)',  # "benefits of [topic]"
            r'advantages of (.+?)(?:\s+for|\.|\n)'  # "advantages of [topic]"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) < 100:  # Reasonable title length
                    return title.title()
        
        # Generate title from first sentence
        first_sentence = content.split('.')[0].strip()
        if 'RAG systems' in first_sentence:
            return "The Benefits of RAG Systems for Modern Businesses"
        elif 'benefits' in first_sentence.lower():
            return "Key Benefits and Advantages"
        
        return "Analysis and Insights"
    
    def _identify_essay_sections(self, content: str) -> Dict[str, str]:
        """Identify introduction, body, and conclusion sections."""
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        total_sentences = len(sentences)
        
        sections = {}
        
        if total_sentences <= 3:
            # Very short content - treat as single body
            sections['body'] = content
        elif total_sentences <= 8:
            # Short essay - intro + body + conclusion
            sections['introduction'] = sentences[0]
            sections['body'] = ' '.join(sentences[1:-1])
            sections['conclusion'] = sentences[-1]
        else:
            # Longer essay - more structured approach
            intro_end = max(1, total_sentences // 6)  # ~15% for intro
            conclusion_start = total_sentences - max(1, total_sentences // 8)  # ~12% for conclusion
            
            sections['introduction'] = ' '.join(sentences[:intro_end])
            sections['body'] = ' '.join(sentences[intro_end:conclusion_start])
            sections['conclusion'] = ' '.join(sentences[conclusion_start:])
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split body text into well-structured paragraphs."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        paragraphs = []
        current_paragraph = []
        sentences_per_paragraph = max(3, len(sentences) // 4)  # Aim for 3-4 paragraphs
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            if len(current_paragraph) >= sentences_per_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def _generate_paragraph_heading(self, paragraph: str, index: int) -> str:
        """Generate appropriate heading for essay paragraph."""
        # Extract key themes from paragraph
        paragraph_lower = paragraph.lower()
        
        # Common essay section patterns
        if 'benefit' in paragraph_lower or 'advantage' in paragraph_lower:
            if 'security' in paragraph_lower or 'privacy' in paragraph_lower:
                return "Security and Privacy Benefits"
            elif 'cost' in paragraph_lower or 'efficiency' in paragraph_lower:
                return "Cost-Effectiveness and Efficiency"
            elif 'knowledge' in paragraph_lower or 'information' in paragraph_lower:
                return "Enhanced Knowledge Management"
            else:
                return f"Strategic Advantage {index}"
        elif 'implementation' in paragraph_lower or 'technical' in paragraph_lower:
            return "Implementation Considerations"
        elif 'challenge' in paragraph_lower or 'limitation' in paragraph_lower:
            return "Challenges and Limitations"
        elif 'future' in paragraph_lower or 'conclusion' in paragraph_lower:
            return "Future Outlook"
        else:
            return f"Analysis Point {index}"
    
    def _format_blog(self, content: str, requirements: Dict[str, any]) -> str:
        """Format content as an engaging blog post with catchy title and subheadings."""
        title = self._extract_blog_title(content)
        paragraphs = self._split_into_paragraphs(content)
        
        formatted_parts = []
        
        # Add catchy title
        if title:
            formatted_parts.append(f"# {title}\n")
        
        # Add engaging intro
        if paragraphs:
            formatted_parts.append(f"{paragraphs[0]}\n")
            
            # Add body with engaging subheadings
            for i, paragraph in enumerate(paragraphs[1:], 1):
                if len(paragraphs) > 3:
                    heading = self._generate_blog_heading(paragraph, i)
                    formatted_parts.append(f"## {heading}\n\n{paragraph}\n")
                else:
                    formatted_parts.append(f"{paragraph}\n")
        
        return '\n'.join(formatted_parts)
    
    def _format_article(self, content: str, requirements: Dict[str, any]) -> str:
        """Format content as a professional article with clear structure."""
        title = self._extract_article_title(content)
        sections = self._identify_article_sections(content)
        
        formatted_parts = []
        
        # Add professional title
        if title:
            formatted_parts.append(f"# {title}\n")
        
        # Add lead paragraph
        if sections.get('lead'):
            formatted_parts.append(f"{sections['lead']}\n")
        
        # Add body sections
        if sections.get('body'):
            body_paragraphs = self._split_into_paragraphs(sections['body'])
            for i, paragraph in enumerate(body_paragraphs, 1):
                if len(body_paragraphs) > 2:
                    heading = self._generate_article_heading(paragraph, i)
                    formatted_parts.append(f"## {heading}\n\n{paragraph}\n")
                else:
                    formatted_parts.append(f"{paragraph}\n")
        
        return '\n'.join(formatted_parts)
    
    def _extract_blog_title(self, content: str) -> str:
        """Generate catchy blog title."""
        if 'RAG systems' in content:
            return "Why RAG Systems Are Game-Changers for Your Business"
        elif 'benefits' in content.lower():
            return "The Ultimate Guide to Business Benefits"
        return "Insights and Analysis"
    
    def _extract_article_title(self, content: str) -> str:
        """Generate professional article title."""
        if 'RAG systems' in content:
            return "RAG Systems: Transforming Business Intelligence and Knowledge Management"
        elif 'benefits' in content.lower():
            return "Comprehensive Analysis of Key Business Benefits"
        return "Professional Analysis and Insights"
    
    def _identify_article_sections(self, content: str) -> Dict[str, str]:
        """Identify article sections (lead + body)."""
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        
        if len(sentences) <= 3:
            return {'body': content}
        
        # First 1-2 sentences as lead
        lead_end = min(2, len(sentences) // 4)
        
        return {
            'lead': ' '.join(sentences[:lead_end]),
            'body': ' '.join(sentences[lead_end:])
        }
    
    def _generate_blog_heading(self, paragraph: str, index: int) -> str:
        """Generate engaging blog subheadings."""
        paragraph_lower = paragraph.lower()
        
        if 'benefit' in paragraph_lower:
            return "🚀 Key Benefits You Can't Ignore"
        elif 'security' in paragraph_lower:
            return "🔒 Security That Actually Works"
        elif 'cost' in paragraph_lower:
            return "💰 Save Money While Boosting Performance"
        elif 'implementation' in paragraph_lower:
            return "⚡ Getting Started: Implementation Tips"
        else:
            return f"💡 Important Insight #{index}"
    
    def _generate_article_heading(self, paragraph: str, index: int) -> str:
        """Generate professional article subheadings."""
        paragraph_lower = paragraph.lower()
        
        if 'benefit' in paragraph_lower or 'advantage' in paragraph_lower:
            return "Strategic Advantages and Benefits"
        elif 'security' in paragraph_lower:
            return "Security and Compliance Considerations"
        elif 'cost' in paragraph_lower or 'efficiency' in paragraph_lower:
            return "Economic Impact and Efficiency Gains"
        elif 'implementation' in paragraph_lower:
            return "Implementation Strategy and Best Practices"
        else:
            return f"Key Consideration {index}"
    
    def _generate_structure_template(self, prompt: str) -> Dict[str, any]:
        """Generate detailed structural template using LLM."""
        structure_prompt = f"""Analyze this user request and create a detailed structure template for the response:

"{prompt}"

Return ONLY valid JSON with this exact structure:
{{
  "title": "Specific, engaging title for this content",
  "sections": [
    {{"heading": "Section name", "purpose": "What this section covers", "weight": 25}},
    {{"heading": "Another section", "purpose": "Content description", "weight": 35}}
  ],
  "style": "professional|academic|conversational|technical",
  "format_type": "essay|article|blog|report|guide"
}}

Ensure section weights sum to 100. Create 3-5 sections appropriate for the request."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": structure_prompt}]
            )
            
            raw_response = response.content[0].text.strip()
            print(f"Raw template response: {raw_response}")  # Debug
            
            # Clean JSON response
            if raw_response.startswith('```json'):
                raw_response = raw_response.replace('```json', '').replace('```', '').strip()
            
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response was: {raw_response}")
            raise
        except Exception as e:
            print(f"Template generation error: {e}")
            raise
    
    def _format_with_template(self, content: str, template: Dict[str, any]) -> str:
        """Format content according to LLM-generated template."""
        print(f"Formatting with template: {template}")
        
        # Split content into sentences for distribution
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        total_sentences = len(sentences)
        print(f"Total sentences: {total_sentences}")
        
        formatted_parts = []
        
        # Add title
        if template.get('title'):
            formatted_parts.append(f"# {template['title']}\n")
            print(f"Added title: {template['title']}")
        
        # Distribute content across sections
        sections = template.get('sections', [])
        sentence_index = 0
        
        for i, section in enumerate(sections):
            heading = section.get('heading', 'Section')
            weight = section.get('weight', 100 // len(sections))
            
            # Calculate sentences for this section
            section_sentences = max(1, int(total_sentences * weight / 100))
            end_index = min(sentence_index + section_sentences, total_sentences)
            
            print(f"Section {i+1}: {heading}, sentences {sentence_index}-{end_index}")
            
            if sentence_index < total_sentences:
                section_content = ' '.join(sentences[sentence_index:end_index])
                formatted_parts.append(f"## {heading}\n\n{section_content}\n")
                sentence_index = end_index
        
        # Add any remaining content
        if sentence_index < total_sentences:
            remaining = ' '.join(sentences[sentence_index:])
            formatted_parts.append(f"## Additional Information\n\n{remaining}")
            print(f"Added remaining content: {len(remaining)} chars")
        
        result = '\n'.join(formatted_parts)
        print(f"Final formatted length: {len(result)} chars")
        return result
    
    def _format_list(self, content: str) -> str:
        """Format content as a bulleted list."""
        # Split content into logical points
        points = re.split(r'[.!?]\s+(?=[A-Z])', content)
        
        formatted_points = []
        for point in points:
            point = point.strip()
            if point and not point.startswith('•'):
                formatted_points.append(f"• {point}")
        
        return '\n\n'.join(formatted_points)
    
    def _format_report(self, content: str) -> str:
        """Format content as a structured report with executive summary, analysis, and recommendations."""
        if '##' in content or '#' in content:
            return content  # Already formatted
        
        # Split content into logical sections
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        total_sentences = len(sentences)
        
        if total_sentences < 6:
            return f"## Report\n\n{content}"
        
        # Structure: 20% summary, 60% analysis, 20% recommendations
        summary_end = max(2, total_sentences // 5)
        analysis_end = total_sentences - max(2, total_sentences // 5)
        
        summary = ' '.join(sentences[:summary_end])
        analysis = ' '.join(sentences[summary_end:analysis_end])
        recommendations = ' '.join(sentences[analysis_end:])
        
        formatted = f"## Executive Summary\n\n{summary}\n\n"
        formatted += f"## Detailed Analysis\n\n{analysis}\n\n"
        formatted += f"## Recommendations\n\n{recommendations}"
        
        return formatted
    
    def _format_outline(self, content: str) -> str:
        """Format content as a hierarchical outline."""
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                if not line.startswith(('I.', 'A.', '1.', '•', '-')):
                    formatted_lines.append(f"• {line}")
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_steps(self, content: str) -> str:
        """Format content as numbered steps."""
        # Split into logical steps
        steps = re.split(r'[.!?]\s+(?=[A-Z])', content)
        
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            step = step.strip()
            if step:
                formatted_steps.append(f"{i}. {step}")
        
        return '\n\n'.join(formatted_steps)
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract a suitable title from content."""
        first_sentence = content.split('.')[0]
        if len(first_sentence) < 100:
            # Clean up the title
            title = re.sub(r'^(Here\'s|This is|The following is)\s+', '', first_sentence, flags=re.IGNORECASE)
            return title.strip()
        return None