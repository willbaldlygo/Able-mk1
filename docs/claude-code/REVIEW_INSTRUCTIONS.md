# Comprehensive Code Review Instructions

## Overview

This document provides step-by-step instructions for conducting a thorough review of the Able mk I codebase to identify redundant code, potential issues, and optimization opportunities.

## Prerequisites

Before starting the review:
1. Ensure you have read `.claude/skills/able-project-review/SKILL.md`
2. Review `CLAUDE.md` for project architecture overview
3. Check `README.md` for current feature documentation
4. Reference `docs/ARCHITECTURE_REFERENCE.md` for quick architecture lookup

## Review Process

### Phase 1: Initial Scanning (15-20 minutes)

**Objective:** Get a high-level understanding of the codebase structure and identify obvious issues.

1. **Count files by type:**
   ```bash
   find . -name "*.py" -not -path "./archive/*" -not -path "*/venv/*" | wc -l
   find . -name "test_*.py" -not -path "./archive/*" | wc -l
   find . -name "*.js" -not -path "*/node_modules/*" -not -path "./archive/*" | wc -l
   ```

2. **Identify large files** (>500 lines):
   ```bash
   find backend -name "*.py" -exec wc -l {} \; | sort -rn | head -20
   ```

3. **Check for obvious duplicates:**
   - Multiple implementations of same functionality
   - Similar file names (e.g., `service.py` vs `service_old.py`)
   - Backup files with `.bak`, `_backup`, `_old` suffixes

4. **Scan test files:**
   - Open each `test_*.py` file in root
   - Check if they have proper pytest structure or are just scripts
   - Look for assertions, fixtures, test classes

### Phase 2: Service Layer Analysis (20-30 minutes)

**Objective:** Verify all backend services are properly integrated and functional.

1. **List all services:**
   ```bash
   ls -la backend/services/
   ```

2. **For each service, verify:**
   - Is it imported in `backend/main.py`?
   - Does it have corresponding API endpoints?
   - Are there any unused imports?
   - Is error handling comprehensive?

3. **Check service dependencies:**
   - Run: `grep -r "from services" backend/`
   - Create dependency map: which services depend on which?
   - Identify circular dependencies

4. **Validate MCP integration:**
   - Check `backend/mcp/` has all three servers: filesystem, git, sqlite
   - Verify MCP endpoints exist in `main.py`
   - Confirm session management is secure (UUID-based)

### Phase 3: Dead Code Detection (30-40 minutes)

**Objective:** Find code that's no longer used or referenced.

1. **Find commented-out code blocks:**
   ```bash
   grep -r "^#.*def \|^#.*class " backend/ | grep -v "__pycache__"
   ```

2. **Detect unused imports:**
   - Use tool like `pylint` or manual inspection
   - Focus on heavy imports (pandas, matplotlib, torch)
   - Check if imports are actually used in the file

3. **Search for dead functions:**
   - List all function definitions: `grep -r "^def " backend/`
   - For suspicious ones, search for their usage across codebase
   - Flag functions with no callers

4. **Check for TODO/FIXME/DEBUG markers:**
   ```bash
   grep -rn "TODO\|FIXME\|DEBUG\|XXX\|HACK" backend/ new-ui/
   ```

### Phase 4: Security Audit (15-20 minutes)

**Objective:** Identify potential security vulnerabilities.

1. **Hardcoded secrets check:**
   ```bash
   grep -rn "api_key\|password\|secret\|token" backend/ --include="*.py" | grep -v ".env"
   ```

2. **Path traversal vulnerabilities:**
   - Review `backend/mcp/filesystem_server.py`
   - Check for proper path validation before file operations
   - Ensure paths are restricted to allowed directories

3. **SQL injection risks:**
   - Review `backend/mcp/sqlite_server.py`
   - Verify parameterized queries, not string concatenation
   - Check user input sanitization

4. **File upload validation:**
   - Review `/upload` and `/upload/multimodal` endpoints
   - Verify file size limits
   - Check allowed file type validation
   - Ensure secure file storage paths

### Phase 5: Performance Analysis (15-20 minutes)

**Objective:** Find performance bottlenecks and optimization opportunities.

1. **Check metadata storage size:**
   ```bash
   ls -lh data/document_metadata.json
   ```
   Should be under 10KB for small document sets. If larger, investigate.

2. **Identify synchronous blocking calls:**
   - Search for: `requests.get`, `requests.post` without async
   - Check LLava calls in `multimodal_service.py`
   - Look for file I/O without streaming

3. **Database operation patterns:**
   - Check ChromaDB query optimization
   - Verify BM25 index is cached, not rebuilt each search
   - Look for N+1 query patterns

4. **Memory usage patterns:**
   - Check for loading entire files into memory
   - Look for temporary file cleanup
   - Verify image processing doesn't leak memory

### Phase 6: Dependency Audit (10-15 minutes)

**Objective:** Ensure all dependencies are necessary, secure, and up-to-date.

1. **List all dependencies:**
   ```bash
   cat backend/requirements.txt
   ```

2. **Check for unused packages:**
   - For each package, search for imports: `grep -r "import package_name" backend/`
   - Flag packages with no imports

3. **Security vulnerabilities:**
   ```bash
   pip list --outdated
   pip-audit  # if available
   ```

4. **Version pinning:**
   - Ensure all packages have version pins (==X.Y.Z)
   - Check for overly restrictive pins (==) vs flexible (~=)

### Phase 7: Frontend Review (10-15 minutes)

**Objective:** Verify frontend code quality and integration.

1. **Check for old React references:**
   ```bash
   grep -rn "frontend/" new-ui/ *.py *.sh
   ```
   Should find none (React frontend is archived).

2. **Validate API integration:**
   - Review `new-ui/script.js`
   - Check all fetch calls have proper error handling
   - Verify endpoint URLs match backend routes

3. **Code quality:**
   - Check for console.log statements (should be removed or conditional)
   - Verify proper event listener cleanup
   - Look for memory leaks (event listeners not removed)

### Phase 8: Documentation Consistency (10 minutes)

**Objective:** Ensure documentation matches implementation.

1. **README.md vs. actual features:**
   - For each feature listed in README, verify it exists
   - Check version numbers are current
   - Validate command examples work

2. **CLAUDE.md accuracy:**
   - Compare endpoint list with `backend/main.py` routes
   - Verify configuration examples match `.env.example`
   - Check architecture descriptions match current code

3. **Code comments:**
   - Look for outdated comments referencing old architecture
   - Check docstrings match function signatures
   - Verify complex logic has explanatory comments

## Deliverables

Create `REVIEW_RESULTS.md` with:

1. **Executive Summary**
   - Total files reviewed
   - Critical issues count
   - Files recommended for archival
   - Overall code health score (1-10)

2. **Detailed Findings by Category**
   - Critical Issues (with file:line references)
   - Dead Code (with evidence of non-use)
   - Security Concerns (with risk levels)
   - Performance Opportunities (with estimated impact)
   - Dependency Issues (with recommendations)

3. **Archival Candidates**
   - List of files safe to move to `archive/`
   - Rationale for each
   - Any dependencies to consider

4. **Action Items**
   - Prioritized list of fixes (High/Medium/Low)
   - Estimated effort for each
   - Suggested order of execution

5. **Metrics**
   - Lines of code breakdown
   - Test coverage estimates
   - Code duplication percentage
   - Documentation coverage

## Tips for Efficient Review

- **Use grep strategically:** Don't read every file - search for patterns
- **Trust git history:** Use `git log` and `git blame` to understand why code exists
- **Focus on high-impact areas:** Services, API endpoints, security-critical code
- **Batch similar issues:** Group all "unused import" findings together
- **Be specific:** Always include file paths and line numbers
- **Validate before flagging:** Confirm dead code truly has no callers
- **Consider impact:** Prioritize findings by risk/benefit

## Next Steps After Review

Once review is complete:
1. Share `REVIEW_RESULTS.md` for discussion
2. Create GitHub issues for critical fixes
3. Plan archival strategy for obsolete code
4. Schedule refactoring sprints for optimization opportunities
5. Update documentation to reflect current state
