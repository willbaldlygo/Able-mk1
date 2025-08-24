"""MCP (Model Context Protocol) integration service for Able chat system."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from models import MCPToolResult, MCPSession, MCPConfig, SourceInfo, MCPEnhancedSourceInfo

logger = logging.getLogger(__name__)

class MCPIntegrationService:
    """Service for integrating MCP tools with Able's chat system."""
    
    def __init__(self):
        """Initialize MCP integration service."""
        self.tool_formatters = {
            # Filesystem tools
            "filesystem_read": self._format_filesystem_read,
            "filesystem_write": self._format_filesystem_write,
            "filesystem_list": self._format_filesystem_list,
            
            # Git tools
            "git_status": self._format_git_status,
            "git_log": self._format_git_log,
            "git_diff": self._format_git_diff,
            "git_show": self._format_git_show,
            
            # SQLite tools
            "sqlite_query": self._format_sqlite_query,
            "sqlite_schema": self._format_sqlite_schema,
            "sqlite_list_tables": self._format_sqlite_list_tables
        }
    
    def format_mcp_results(self, mcp_results: List[MCPToolResult]) -> str:
        """Format MCP tool results for inclusion in chat responses."""
        if not mcp_results:
            return ""
        
        formatted_sections = []
        
        for result in mcp_results:
            if not result.success:
                # Format error results
                error_section = f"**{result.tool_name} Error:**\n{result.error or 'Unknown error occurred'}"
                formatted_sections.append(error_section)
                continue
            
            # Use specific formatter if available
            formatter = self.tool_formatters.get(result.tool_name, self._format_generic_result)
            formatted_content = formatter(result)
            
            if formatted_content:
                formatted_sections.append(formatted_content)
        
        if not formatted_sections:
            return ""
        
        # Combine all sections with clear separation
        mcp_section = "\n\n## MCP Tool Results\n\n" + "\n\n---\n\n".join(formatted_sections)
        return mcp_section
    
    def enhance_sources_with_mcp(
        self, 
        sources: List[SourceInfo], 
        mcp_results: List[MCPToolResult]
    ) -> List[MCPEnhancedSourceInfo]:
        """Enhance source information with MCP tool results."""
        enhanced_sources = []
        
        for source in sources:
            # Convert to enhanced source
            enhanced_source = MCPEnhancedSourceInfo(
                document_id=source.document_id,
                document_name=source.document_name,
                chunk_content=source.chunk_content,
                relevance_score=source.relevance_score,
                mcp_results=[],  # Will be populated based on relevance
                entities=[],  # Placeholder for GraphRAG entities
                relationships=[]  # Placeholder for GraphRAG relationships
            )
            
            # Add relevant MCP results to source
            # This could be enhanced with more sophisticated matching logic
            enhanced_source.mcp_results = [
                result for result in mcp_results 
                if result.success and self._is_result_relevant_to_source(result, source)
            ]
            
            enhanced_sources.append(enhanced_source)
        
        return enhanced_sources
    
    def create_tool_context_for_llm(self, session: MCPSession, query: str) -> str:
        """Create context about available MCP tools for LLM."""
        if not session or not session.enabled or not session.available_tools:
            return ""
        
        tool_descriptions = []
        
        # Group tools by category
        categories = {}
        for tool in session.available_tools:
            category = self._get_tool_category(tool)
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        
        # Format tool information
        context_parts = ["You have access to the following MCP tools:"]
        
        for category, tools in categories.items():
            context_parts.append(f"\n**{category.title()} Tools:**")
            for tool in tools:
                description = self._get_tool_description(tool)
                context_parts.append(f"- {tool}: {description}")
        
        # Add usage guidelines
        context_parts.extend([
            "\nMCP Tool Usage Guidelines:",
            "- Use filesystem tools to access local files and directories",
            "- Use git tools to examine repository history and changes",
            "- Use sqlite tools to query database information",
            "- Always specify the tool name and required parameters clearly",
            "- MCP tools complement document search - use both for comprehensive analysis"
        ])
        
        return "\n".join(context_parts)
    
    def _format_filesystem_read(self, result: MCPToolResult) -> str:
        """Format filesystem read results."""
        content = result.content or ""
        filename = result.metadata.get('file_path', 'Unknown file')
        
        # Truncate very long content
        if len(content) > 1000:
            content = content[:1000] + f"\n... (content truncated, {len(result.content)} total characters)"
        
        return f"**File Content ({filename}):**\n```\n{content}\n```"
    
    def _format_filesystem_write(self, result: MCPToolResult) -> str:
        """Format filesystem write results."""
        filename = result.metadata.get('file_path', 'Unknown file')
        bytes_written = result.metadata.get('bytes_written', 'Unknown')
        
        return f"**File Write Success:** {filename} ({bytes_written} bytes written)"
    
    def _format_filesystem_list(self, result: MCPToolResult) -> str:
        """Format filesystem list results."""
        directory = result.metadata.get('directory_path', 'Unknown directory')
        items = result.content.split('\n') if result.content else []
        
        if len(items) > 20:
            visible_items = items[:20]
            truncated_count = len(items) - 20
            content = "\n".join(visible_items) + f"\n... ({truncated_count} more items)"
        else:
            content = "\n".join(items)
        
        return f"**Directory Listing ({directory}):**\n```\n{content}\n```"
    
    def _format_git_status(self, result: MCPToolResult) -> str:
        """Format git status results."""
        repo = result.metadata.get('repository_path', 'Unknown repository')
        return f"**Git Status ({repo}):**\n```\n{result.content}\n```"
    
    def _format_git_log(self, result: MCPToolResult) -> str:
        """Format git log results."""
        repo = result.metadata.get('repository_path', 'Unknown repository')
        limit = result.metadata.get('limit', 'All')
        return f"**Git Log ({repo}, limit: {limit}):**\n```\n{result.content}\n```"
    
    def _format_git_diff(self, result: MCPToolResult) -> str:
        """Format git diff results."""
        repo = result.metadata.get('repository_path', 'Unknown repository')
        commit_ref = result.metadata.get('commit_ref', 'HEAD')
        
        # Truncate very long diffs
        content = result.content or ""
        if len(content) > 2000:
            content = content[:2000] + "\n... (diff truncated for display)"
        
        return f"**Git Diff ({repo}, ref: {commit_ref}):**\n```diff\n{content}\n```"
    
    def _format_git_show(self, result: MCPToolResult) -> str:
        """Format git show results."""
        repo = result.metadata.get('repository_path', 'Unknown repository')
        object_ref = result.metadata.get('object_ref', 'Unknown object')
        return f"**Git Show ({repo}, object: {object_ref}):**\n```\n{result.content}\n```"
    
    def _format_sqlite_query(self, result: MCPToolResult) -> str:
        """Format SQLite query results."""
        database = result.metadata.get('database', 'Unknown database')
        query = result.metadata.get('query', 'Unknown query')
        
        # Format as table if possible
        content = result.content or ""
        if '\t' in content or '|' in content:
            # Looks like tabular data
            return f"**SQLite Query Results ({database}):**\nQuery: `{query}`\n```\n{content}\n```"
        else:
            return f"**SQLite Query Results ({database}):**\nQuery: `{query}`\nResult: {content}"
    
    def _format_sqlite_schema(self, result: MCPToolResult) -> str:
        """Format SQLite schema results."""
        database = result.metadata.get('database', 'Unknown database')
        return f"**SQLite Schema ({database}):**\n```sql\n{result.content}\n```"
    
    def _format_sqlite_list_tables(self, result: MCPToolResult) -> str:
        """Format SQLite list tables results."""
        database = result.metadata.get('database', 'Unknown database')
        return f"**SQLite Tables ({database}):**\n{result.content}"
    
    def _format_generic_result(self, result: MCPToolResult) -> str:
        """Format generic MCP tool results."""
        return f"**{result.tool_name} Result:**\n{result.content or 'No content available'}"
    
    def _is_result_relevant_to_source(self, result: MCPToolResult, source: SourceInfo) -> bool:
        """Check if MCP result is relevant to a document source."""
        # Simple relevance check - could be enhanced with more sophisticated logic
        if not result.success or not result.content:
            return False
        
        # Check if result mentions the document name or ID
        result_text = (result.content or "").lower()
        source_indicators = [
            source.document_name.lower(),
            source.document_id.lower(),
            # Could add more sophisticated matching
        ]
        
        return any(indicator in result_text for indicator in source_indicators)
    
    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool name."""
        if tool_name.startswith("filesystem_"):
            return "filesystem"
        elif tool_name.startswith("git_"):
            return "git"
        elif tool_name.startswith("sqlite_"):
            return "sqlite"
        else:
            return "other"
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool name."""
        descriptions = {
            "filesystem_read": "Read file contents",
            "filesystem_write": "Write content to files",
            "filesystem_list": "List directory contents",
            "git_status": "Get repository status",
            "git_log": "View commit history",
            "git_diff": "Show changes/differences",
            "git_show": "Display specific commits or objects",
            "sqlite_query": "Execute SQL queries",
            "sqlite_schema": "View database schema",
            "sqlite_list_tables": "List database tables"
        }
        return descriptions.get(tool_name, "Execute tool operation")