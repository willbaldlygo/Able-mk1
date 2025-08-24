# Able3 Archive Directory

This directory contains archived components from the Able3 project that are no longer actively used but preserved for historical reference and potential recovery.

**Archive Created:** August 24, 2025
**Full Backup Location:** `/Users/will/AVI BUILD/Able3_Main_WithVoiceMode_BACKUP_20250824_103200`

## Archive Structure

### archive/mothballed-react-frontend/
**Contents:** Complete React frontend application (formerly on port 3000)
**Status:** Mothballed - replaced by new-ui (port 3001)
**Contains:**
- `frontend/` - Complete React 18 application with Tailwind CSS
- Components: App.js, ChatInterface.js, DocumentManager.js, etc.
- Build tools: package.json, tailwind.config.js
- Built with react-scripts

### archive/development-coordination/
**Contents:** Multi-agent development coordination system
**Status:** Development tool - no longer needed for maintenance
**Contains:**
- `coordination-setup.sh` - Setup script for agent coordination
- `coordination/` - Task files and agent specifications
- `agent-commands.md` - Command reference for different specialist agents
- Task files for backend, frontend, devops, and AI optimization agents

### archive/build-logs/
**Contents:** Historical build and implementation logs
**Status:** Documentation of completed work
**Contains:**
- `BUILD_LOG_NEW_UI_IMPLEMENTATION.md` - New UI implementation history
- `BUILD_LOG_STORAGE_CONSISTENCY_FIX.md` - Storage consistency fix log
- `SESSION_LOG.md` - Development session logs
- `DOCK_APP_README.md` - Dock app documentation (may be redundant)

### archive/prototypes/
**Contents:** UI prototypes and mockup assets
**Status:** Development artifacts - no longer needed
**Contains:**
- `vintage_ibm_ui.html` - Vintage UI style prototype
- `test_thinking.html` - Test/demo file from new-ui development
- `ABLE UI ASSETS/` - UI mockups and design assets (SVG files, examples)

### archive/testing-scripts/
**Contents:** Development and testing utilities
**Status:** Development tools - archived for reference
**Contains:**
- `test_dock_app.py` - Dock app testing script
- `test_port_management.py` - Port management testing
- `simple_cleanup.py` - Development cleanup utility

## Recovery Instructions

If you need to recover any archived component:

1. **Full Recovery:** Restore from the complete backup:
   ```bash
   cp -r "/Users/will/AVI BUILD/Able3_Main_WithVoiceMode_BACKUP_20250824_103200" "/Users/will/AVI BUILD/Able3_Main_WithVoiceMode_Recovered"
   ```

2. **Partial Recovery:** Copy specific components from archive back to project root:
   ```bash
   # Example: Restore React frontend
   cp -r archive/mothballed-react-frontend/frontend .
   ```

## Current Active Components

After archival, the active Able3 components are:
- `new-ui/` - Primary web interface (port 3001)
- `backend/` - FastAPI server (port 8000)
- `launch_able.py` - Recommended startup script
- `start_able.py` + `start_able.sh` - Advanced startup options
- `create_dock_app.py` + `install_dock_app.sh` - macOS dock app
- `port_manager.py` - Port management utilities

## Files Deleted Safely

The following files were permanently removed as they were temporary/log files:
- `able_dock_launch.log` - Runtime log
- `able_simple_dock.log` - Runtime log  
- `backend/services/web_scraper_service.py.backup` - Backup file
- `data/metadata_backup.json` - Backup file

## Files Kept for Monitoring

These files remain in the project root but may be archived in future cleanup:
- `create_portable_dock_app.py` - Alternative dock app implementation
- `create_simple_dock_app.py` - Simple dock app implementation
- `ABLE_WEBUI_INTEGRATION_GUIDE.md` - Integration documentation (may be obsolete)

## Notes

- All archived components were fully functional when archived
- The React frontend (port 3000) was officially mothballed per CLAUDE.md
- The coordination system was a development tool and is no longer needed
- Archive preserves complete development history and allows recovery if needed