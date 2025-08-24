"""
MCP (Model Context Protocol) Service for Able
Provides session-based MCP server management with security boundaries
"""

import asyncio
import logging
import os
import subprocess
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from config import config

logger = logging.getLogger(__name__)

class MCPSession:
    """Represents an MCP session with active servers"""
    
    def __init__(self, session_id: str, root_path: str):
        self.session_id = session_id
        self.root_path = Path(root_path).resolve()
        self.servers: Dict[str, subprocess.Popen] = {}
        self.server_configs: Dict[str, Dict] = {}
        self.active = True
        
    async def start_server(self, server_type: str, config: Dict[str, Any]) -> bool:
        """Start an MCP server of the specified type"""
        if server_type in self.servers:
            logger.warning(f"Server {server_type} already running in session {self.session_id}")
            return False
            
        try:
            if server_type == "filesystem":
                return await self._start_filesystem_server(config)
            elif server_type == "git":
                return await self._start_git_server(config)
            elif server_type == "sqlite":
                return await self._start_sqlite_server(config)
            else:
                logger.error(f"Unknown server type: {server_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start {server_type} server: {e}")
            return False
    
    async def _start_filesystem_server(self, config: Dict[str, Any]) -> bool:
        """Start filesystem MCP server"""
        from mcp.filesystem_server import start_filesystem_server
        
        # Validate root path is within allowed boundaries
        fs_root = Path(config.get('root_path', self.root_path)).resolve()
        if not self._is_path_allowed(fs_root):
            logger.error(f"Filesystem root {fs_root} not within allowed boundaries")
            return False
            
        server_process = await start_filesystem_server(fs_root)
        if server_process:
            self.servers["filesystem"] = server_process
            self.server_configs["filesystem"] = config
            logger.info(f"Filesystem server started for session {self.session_id}")
            return True
        return False
    
    async def _start_git_server(self, config: Dict[str, Any]) -> bool:
        """Start Git MCP server"""
        from mcp.git_server import start_git_server
        
        # Validate git repositories are within allowed boundaries
        repo_paths = config.get('allowed_repos', [self.root_path])
        for repo_path in repo_paths:
            if not self._is_path_allowed(Path(repo_path).resolve()):
                logger.error(f"Git repository {repo_path} not within allowed boundaries")
                return False
        
        server_process = await start_git_server(repo_paths)
        if server_process:
            self.servers["git"] = server_process
            self.server_configs["git"] = config
            logger.info(f"Git server started for session {self.session_id}")
            return True
        return False
    
    async def _start_sqlite_server(self, config: Dict[str, Any]) -> bool:
        """Start SQLite MCP server"""
        from mcp.sqlite_server import start_sqlite_server
        
        # Validate database paths are within allowed boundaries
        db_paths = config.get('allowed_dbs', [])
        for db_path in db_paths:
            if not self._is_path_allowed(Path(db_path).resolve()):
                logger.error(f"Database {db_path} not within allowed boundaries")
                return False
        
        server_process = await start_sqlite_server(db_paths, self.root_path)
        if server_process:
            self.servers["sqlite"] = server_process
            self.server_configs["sqlite"] = config
            logger.info(f"SQLite server started for session {self.session_id}")
            return True
        return False
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within allowed boundaries"""
        try:
            path.resolve().relative_to(self.root_path)
            return True
        except ValueError:
            # Also allow if path is the root path itself
            if path.resolve() == self.root_path:
                return True
            return False
    
    async def stop_server(self, server_type: str) -> bool:
        """Stop a specific MCP server"""
        if server_type not in self.servers:
            logger.warning(f"Server {server_type} not running in session {self.session_id}")
            return False
        
        try:
            server_process = self.servers[server_type]
            server_process.terminate()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process(server_process)), 
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Force killing {server_type} server")
                server_process.kill()
            
            del self.servers[server_type]
            del self.server_configs[server_type]
            logger.info(f"{server_type} server stopped for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {server_type} server: {e}")
            return False
    
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for process to terminate"""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def stop_all_servers(self):
        """Stop all servers in this session"""
        server_types = list(self.servers.keys())
        for server_type in server_types:
            await self.stop_server(server_type)
        
        self.active = False
        logger.info(f"All servers stopped for session {self.session_id}")
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers in this session"""
        status = {
            "session_id": self.session_id,
            "active": self.active,
            "root_path": str(self.root_path),
            "servers": {}
        }
        
        for server_type, process in self.servers.items():
            status["servers"][server_type] = {
                "running": process.poll() is None,
                "pid": process.pid,
                "config": self.server_configs.get(server_type, {})
            }
        
        return status


class MCPManager:
    """Manages MCP sessions and server lifecycle"""
    
    def __init__(self):
        self.sessions: Dict[str, MCPSession] = {}
        self.default_root = Path("../data").resolve()
        
    def create_session(self, root_path: Optional[str] = None) -> str:
        """Create a new MCP session"""
        session_id = str(uuid.uuid4())
        
        if root_path is None:
            root_path = str(self.default_root)
        
        # Validate root path is within allowed boundaries
        root_path_resolved = Path(root_path).resolve()
        if not self._is_root_path_allowed(root_path_resolved):
            raise ValueError(f"Root path {root_path_resolved} not within allowed boundaries")
        
        session = MCPSession(session_id, str(root_path_resolved))
        self.sessions[session_id] = session
        
        logger.info(f"Created MCP session {session_id} with root {root_path_resolved}")
        return session_id
    
    def _is_root_path_allowed(self, root_path: Path) -> bool:
        """Check if a root path is allowed"""
        # Allow paths within the project directory or data directory
        project_root = Path(__file__).parent.parent.resolve()  # backend parent = project root
        
        try:
            root_path.relative_to(project_root)
            return True
        except ValueError:
            pass
            
        # Also allow the data directory specifically
        try:
            root_path.relative_to(self.default_root)
            return True
        except ValueError:
            pass
            
        return False
    
    async def start_server(self, session_id: str, server_type: str, config: Dict[str, Any]) -> bool:
        """Start an MCP server in a session"""
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.sessions[session_id]
        if not session.active:
            logger.error(f"Session {session_id} is not active")
            return False
        
        return await session.start_server(server_type, config)
    
    async def stop_server(self, session_id: str, server_type: str) -> bool:
        """Stop an MCP server in a session"""
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        return await self.sessions[session_id].stop_server(server_type)
    
    async def stop_session(self, session_id: str) -> bool:
        """Stop all servers and close a session"""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return False
        
        session = self.sessions[session_id]
        await session.stop_all_servers()
        del self.sessions[session_id]
        
        logger.info(f"Stopped and removed session {session_id}")
        return True
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        if session_id not in self.sessions:
            return None
        
        return self.sessions[session_id].get_server_status()
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all sessions"""
        return {
            "total_sessions": len(self.sessions),
            "sessions": {
                session_id: session.get_server_status()
                for session_id, session in self.sessions.items()
            }
        }
    
    async def cleanup_inactive_sessions(self):
        """Clean up inactive sessions"""
        inactive_sessions = []
        
        for session_id, session in self.sessions.items():
            if not session.active:
                inactive_sessions.append(session_id)
            else:
                # Check if any servers have died
                dead_servers = []
                for server_type, process in session.servers.items():
                    if process.poll() is not None:
                        dead_servers.append(server_type)
                
                # Remove dead servers
                for server_type in dead_servers:
                    logger.warning(f"Removing dead {server_type} server from session {session_id}")
                    del session.servers[server_type]
                    del session.server_configs[server_type]
                
                # Mark session as inactive if no servers left
                if not session.servers:
                    session.active = False
                    inactive_sessions.append(session_id)
        
        # Remove inactive sessions
        for session_id in inactive_sessions:
            await self.stop_session(session_id)
        
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
    
    async def shutdown_all(self):
        """Shutdown all sessions and servers"""
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.stop_session(session_id)
        
        logger.info("All MCP sessions shutdown")


# Global MCP manager instance
mcp_manager = MCPManager()

async def get_mcp_manager() -> MCPManager:
    """Get the global MCP manager instance"""
    return mcp_manager