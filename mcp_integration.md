# MCP Integration Implementation Plan

## Status: 80% Complete - Critical Integration Gap

### Agent Work Summary
- **UI Agent**: ✅ Complete - Architecture design validated
- **Frontend Agent**: ✅ Complete - API endpoints implemented  
- **Backend Agent**: ❌ Incomplete - Hit 5-hour limit, SQLite server partial
- **Architect Agent**: ❌ Incomplete - Hit 5-hour limit, identified security gap

### Critical Issue
`main.py` uses placeholder MCP session management instead of actual `MCPManager`, bypassing security controls.

## Implementation Tasks

### Phase 1: Fix Backend Integration (CRITICAL)

#### 1. Update main.py MCP Endpoints
**File**: `backend/main.py`

Replace placeholder logic with `mcp_manager` calls:

```python
# Current (INSECURE):
mcp_session: Optional[MCPSession] = None

# Fixed (SECURE):
from services.mcp_service import mcp_manager
current_mcp_session_id: Optional[str] = None
```

**Endpoints to fix**:
- `/mcp/toggle` - Use `mcp_manager.create_session()`
- `/mcp/config` - Use `mcp_manager.start_server()`  
- `/mcp/status` - Use `mcp_manager.get_session_status()`
- `/mcp/tools` - Reflect actual server capabilities

#### 2. Complete SQLite Server
**File**: `backend/mcp/sqlite_server.py`
**Status**: 892 lines implemented, needs completion

#### 3. Add MCP Dependencies
**File**: `backend/requirements.txt`
Add: `mcp>=1.0.0` (or appropriate MCP library)

### Phase 2: Frontend UI Implementation

#### Files to Create/Modify:
- `new-ui/index.html` - MCP toggle button, status indicator
- `new-ui/script.js` - MCP API calls, configuration modal
- `new-ui/styles.css` - MCP UI styling

#### UI Components Needed:
1. Header MCP toggle button
2. MCP configuration modal (filesystem/git/sqlite paths)
3. MCP status indicator
4. Enhanced chat to show MCP tool results

### Phase 3: Configuration

#### Environment Variables (.env):
```bash
MCP_ENABLED=true
MCP_DEFAULT_ROOT=/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data
MCP_SESSION_TIMEOUT=3600
```

## Security Architecture (Implemented)

### MCPManager Security Features:
- ✅ Path validation within session boundaries
- ✅ Process isolation with UUID sessions
- ✅ Graceful shutdown with force-kill fallback
- ✅ Restricted filesystem/git/sqlite access

### Current Security Gap:
- ❌ `main.py` bypasses MCPManager security
- ❌ No path validation in placeholder code

## Testing Checklist

- [ ] MCP enable/disable via API
- [ ] Filesystem tools with path restrictions
- [ ] Git tools with repo restrictions  
- [ ] SQLite tools with database restrictions
- [ ] UI MCP controls functional
- [ ] Chat shows MCP tool results
- [ ] Security boundaries enforced
- [ ] Graceful error handling

## File Structure

```
backend/
├── main.py                    # ❌ NEEDS INTEGRATION FIX
├── services/
│   ├── mcp_service.py        # ✅ COMPLETE
│   └── mcp_integration_service.py # ✅ COMPLETE
└── mcp/
    ├── filesystem_server.py  # ✅ COMPLETE
    ├── git_server.py         # ✅ COMPLETE
    └── sqlite_server.py      # ❌ NEEDS COMPLETION

new-ui/
├── index.html                # ❌ NEEDS MCP UI
├── script.js                 # ❌ NEEDS MCP FUNCTIONS
└── styles.css                # ❌ NEEDS MCP STYLES
```

## Priority Order

1. **CRITICAL**: Fix `main.py` integration with MCPManager
2. **HIGH**: Complete SQLite server implementation
3. **MEDIUM**: Add basic MCP UI toggle
4. **LOW**: Full configuration UI and advanced features

## Ready for Implementation

All architecture is validated, services are implemented, security model is designed. Main blocker is connecting the placeholder session management in `main.py` to the actual secure `MCPManager` system.

**Estimated completion time**: 2-3 hours for critical fixes, 4-6 hours for full UI implementation.