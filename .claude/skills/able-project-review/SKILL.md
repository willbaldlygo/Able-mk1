---
name: able-project-review
description: Expert knowledge for reviewing and maintaining the Able mk I multimodal research assistant codebase
---

# Able Project Review Skill

## Project Context

Able mk I is a production-grade multimodal AI research assistant with sophisticated document processing capabilities.

**Tech Stack:**
- Backend: Python 3.11+, FastAPI, ChromaDB, GraphRAG, NetworkX
- Frontend: Vanilla JavaScript (new-ui/), modern HTML/CSS
- AI Models: Anthropic Claude (text), LLava via Ollama (vision)
- Integrations: MCP (Model Context Protocol), GraphRAG 2.6.0

**Key Features:**
- PDF image extraction and analysis using LLava vision model
- Child/parent chunking (700/2000 tokens) with visual context
- Hybrid retrieval: Vector + BM25 lexical + Cross-encoder reranking
- GraphRAG entity extraction and relationship mapping
- MCP integration for filesystem, git, and SQLite operations
- Session management with graceful degradation

## Architecture Patterns to Preserve

### Service-Based Architecture
All core logic lives in `backend/services/`:
- `multimodal_service.py` - LLava integration, image processing
- `document_service.py` - PDF processing, chunking, GraphRAG
- `vector_service.py` - ChromaDB, embeddings, MMR diversity
- `ai_service.py` - Claude API integration
- `graphrag_service.py` - Entity extraction, knowledge graphs
- `mcp_service.py` & `mcp_integration_service.py` - MCP orchestration
- `storage_service.py` - File management, metadata persistence

### Always-On Multimodal
- No toggles for vision processing - it's core functionality
- Graceful degradation when LLava unavailable (metadata-only mode)
- Visual context automatically integrated into search results

### Launch System
- Primary: `launch_able.py` (newer, with port management)
- Legacy: `start_able.py` (keep for compatibility)
- Automatic Ollama startup and model verification

## Review Guidelines

### Critical Files - DO NOT ARCHIVE
1. **Core Services**: Everything in `backend/services/`
2. **MCP Implementation**: All files in `backend/mcp/` (recent August 2025 implementation)
3. **Launch Scripts**: `launch_able.py`, `start_able.py`, `port_manager.py`
4. **Frontend**: All files in `new-ui/` (index.html, script.js, styles.css)
5. **Documentation**: `README.md`, `CLAUDE.md`, `/docs/`
6. **Configuration**: `.env.example`, `backend/requirements.txt`, `backend/config.py`

### Safe to Archive/Delete

**Test Files Without Real Tests:**
- Files named `test_*.py` in root that are just exploration scripts
- Check for: No assertions, no pytest fixtures, just manual testing code
- Example: `test_graphrag.py`, `test_multimodal.py` if they're just exploratory

**Duplicate Implementations:**
- Multiple versions of same functionality
- Old implementations superseded by newer code
- Check git history to confirm obsolescence

**Dead Code Indicators:**
- Large commented-out blocks (>20 lines)
- Unused imports (especially heavy dependencies like pandas, matplotlib)
- Functions/classes with no callers (use grep/search to verify)

**Development Artifacts:**
- Temporary debugging code with TODO/FIXME comments
- Print statements for debugging (should use logging instead)
- Hardcoded test data in production code

### Architecture Validation Checks

**Service Integration:**
```python
# Verify all services are imported and initialized in main.py
- MultimodalService
- DocumentService
- VectorService
- GraphRAGService
- MCPService
- MCPIntegrationService
```

**Endpoint Completeness:**
```
Required multimodal endpoints:
- POST /upload/multimodal
- POST /chat/multimodal
- GET /multimodal/capabilities
- GET /sources/images/{filename}

Required GraphRAG endpoints:
- GET /graph/statistics
- GET /entities/{name}/relationships
- POST /documents/{id}/reprocess-graph

Required MCP endpoints:
- GET /mcp/status
- POST /mcp/toggle
- POST /mcp/config
```

**Configuration Validation:**
```bash
# Check .env has all required keys:
- ANTHROPIC_API_KEY
- OLLAMA_ENABLED=true
- GRAPHRAG_ENABLED=true
- MCP_ENABLED=true
```

### Security Review Checklist

**High Priority:**
1. No hardcoded API keys in code files (check for ANTHROPIC_API_KEY in .py files)
2. Path validation in MCP filesystem operations (check `backend/mcp/filesystem_server.py`)
3. SQL injection prevention in SQLite MCP (check `backend/mcp/sqlite_server.py`)
4. File upload validation (check max size, allowed types in `main.py`)

**Medium Priority:**
5. Session isolation in MCP (UUID-based, no cross-session leaks)
6. CORS configuration appropriate for production
7. Rate limiting on API endpoints (if deployed publicly)

### Performance Red Flags

**Storage Issues:**
- `document_metadata.json` > 50KB (should be ~5.6KB after August 2024 optimization)
- Duplicate chunk content in multiple storage locations
- Full chunk text stored in metadata (should only have references)

**Processing Bottlenecks:**
- Synchronous LLava calls blocking request threads (should be async)
- Missing ChromaDB collection indexes
- No caching for repeated image analysis
- BM25 index rebuild on every search (should be cached)

**Memory Concerns:**
- Loading entire documents into memory instead of streaming
- Not cleaning up temporary image files after processing
- Vector embeddings recomputed unnecessarily

### Code Quality Standards

**Python Best Practices:**
- Type hints on all function signatures
- Docstrings for public functions/classes
- Error handling with specific exceptions (not bare `except:`)
- Logging instead of print statements
- PEP 8 compliance (use black/ruff for formatting)

**API Conventions:**
- Consistent error response format across all endpoints
- Proper HTTP status codes (200, 201, 400, 404, 500)
- Request/response validation with Pydantic models
- Clear error messages for debugging

### Dependency Management

**Check for:**
- Unused packages in `requirements.txt`
- Outdated packages with security vulnerabilities
- Conflicting version requirements
- Missing version pins (should all be pinned for reproducibility)

**Heavy Dependencies to Justify:**
- pandas (only if doing data analysis)
- matplotlib/seaborn (only if generating visualizations)
- torch (only if running local models beyond LLava)

## Common Issues in This Codebase

Based on development history, watch for:

1. **Old React Frontend References**: Already archived, but check for import statements or configuration pointing to old `frontend/` directory

2. **Multiple Launch Mechanisms**: `start_able.py` vs `launch_able.py` - latter is newer and preferred

3. **Test File Confusion**: Many `test_*.py` files are exploratory scripts, not proper unit tests

4. **Port Management**: Old processes may not clean up properly - check for stale port occupancy handling

5. **Model Service Complexity**: If Anthropic is primary provider, Ollama code paths may be underutilized

## Review Output Format

When conducting a review, structure findings as:

```markdown
# Code Review Results - [Date]

## Executive Summary
- X critical issues found
- Y files safe to archive
- Z optimization opportunities

## Critical Issues
### [Issue Name] - CRITICAL
**Location:** `path/to/file.py:line`
**Problem:** [Description]
**Impact:** [Security/Performance/Functionality]
**Recommendation:** [Fix/Archive/Refactor]

## Files to Archive
- `path/to/file.py` - Reason: [Why it's obsolete]

## Optimization Opportunities
- [Description] - Potential improvement: [Impact]

## Security Findings
- [Issue] - Risk level: [High/Medium/Low]

## Dependencies Review
- Unused: [List]
- Outdated: [List with CVEs if any]
- Recommended updates: [List]
```

## How to Use This Skill

This skill activates automatically when you:
- Reference "able" or "Able mk I" in your prompt
- Work with files in this project directory
- Request code review, architecture analysis, or cleanup tasks

Claude will apply these guidelines to ensure consistency with the project's architecture and maintain code quality standards.
