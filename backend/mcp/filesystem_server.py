"""
Filesystem MCP Server for Able
Provides secure file operations within configured boundaries
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class FilesystemMCPServer:
    """MCP server for filesystem operations"""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path.resolve()
        self.tools = [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (relative to root)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (relative to root)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List contents of a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to directory (relative to root, defaults to root)"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to list recursively",
                            "default": False
                        }
                    }
                }
            },
            {
                "name": "create_directory",
                "description": "Create a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to create (relative to root)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "delete_file",
                "description": "Delete a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to delete (relative to root)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "move_file",
                "description": "Move or rename a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source path (relative to root)"
                        },
                        "destination": {
                            "type": "string", 
                            "description": "Destination path (relative to root)"
                        }
                    },
                    "required": ["source", "destination"]
                }
            },
            {
                "name": "search_files",
                "description": "Search for files by name pattern",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "Path to search in (relative to root, defaults to root)"
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "get_file_info",
                "description": "Get file or directory information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to get info for (relative to root)"
                        }
                    },
                    "required": ["path"]
                }
            }
        ]
    
    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path and ensure it's within root boundaries"""
        if not relative_path:
            return self.root_path
        
        # Remove leading slash if present to ensure it's treated as relative
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
        
        full_path = (self.root_path / relative_path).resolve()
        
        # Ensure the path is within our root directory
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            raise ValueError(f"Path {relative_path} is outside allowed root directory")
        
        return full_path
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call and return result"""
        try:
            if tool_name == "read_file":
                return await self._read_file(arguments["path"])
            elif tool_name == "write_file":
                return await self._write_file(arguments["path"], arguments["content"])
            elif tool_name == "list_directory":
                return await self._list_directory(
                    arguments.get("path", ""),
                    arguments.get("recursive", False)
                )
            elif tool_name == "create_directory":
                return await self._create_directory(arguments["path"])
            elif tool_name == "delete_file":
                return await self._delete_file(arguments["path"])
            elif tool_name == "move_file":
                return await self._move_file(arguments["source"], arguments["destination"])
            elif tool_name == "search_files":
                return await self._search_files(
                    arguments["pattern"],
                    arguments.get("path", "")
                )
            elif tool_name == "get_file_info":
                return await self._get_file_info(arguments["path"])
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
            }
    
    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents"""
        file_path = self._resolve_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        try:
            # Try to read as text first
            content = file_path.read_text(encoding='utf-8')
            return {
                "content": [{"type": "text", "text": content}]
            }
        except UnicodeDecodeError:
            # If it's binary, provide basic info
            size = file_path.stat().st_size
            return {
                "content": [{
                    "type": "text", 
                    "text": f"Binary file: {path} ({size} bytes)"
                }]
            }
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        file_path = self._resolve_path(path)
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path.write_text(content, encoding='utf-8')
        
        return {
            "content": [{
                "type": "text", 
                "text": f"Successfully wrote {len(content)} characters to {path}"
            }]
        }
    
    async def _list_directory(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        """List directory contents"""
        dir_path = self._resolve_path(path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        items = []
        
        if recursive:
            # Recursive listing
            for item in dir_path.rglob("*"):
                try:
                    relative_path = item.relative_to(self.root_path)
                    stat = item.stat()
                    items.append({
                        "name": item.name,
                        "path": str(relative_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime
                    })
                except (OSError, ValueError):
                    continue
        else:
            # Non-recursive listing
            try:
                for item in sorted(dir_path.iterdir()):
                    try:
                        relative_path = item.relative_to(self.root_path)
                        stat = item.stat()
                        items.append({
                            "name": item.name,
                            "path": str(relative_path),
                            "type": "directory" if item.is_dir() else "file",
                            "size": stat.st_size if item.is_file() else None,
                            "modified": stat.st_mtime
                        })
                    except (OSError, ValueError):
                        continue
            except PermissionError:
                raise ValueError(f"Permission denied accessing directory: {path}")
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(items, indent=2)
            }]
        }
    
    async def _create_directory(self, path: str) -> Dict[str, Any]:
        """Create a directory"""
        dir_path = self._resolve_path(path)
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully created directory: {path}"
            }]
        }
    
    async def _delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file or directory"""
        file_path = self._resolve_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if file_path.is_dir():
            shutil.rmtree(file_path)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully deleted directory: {path}"
                }]
            }
        else:
            file_path.unlink()
            return {
                "content": [{
                    "type": "text",
                    "text": f"Successfully deleted file: {path}"
                }]
            }
    
    async def _move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move or rename a file"""
        source_path = self._resolve_path(source)
        dest_path = self._resolve_path(destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        # Create parent directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(source_path), str(dest_path))
        
        return {
            "content": [{
                "type": "text",
                "text": f"Successfully moved {source} to {destination}"
            }]
        }
    
    async def _search_files(self, pattern: str, path: str = "") -> Dict[str, Any]:
        """Search for files matching a pattern"""
        search_path = self._resolve_path(path)
        
        if not search_path.exists():
            raise FileNotFoundError(f"Search path not found: {path}")
        
        if not search_path.is_dir():
            raise ValueError(f"Search path is not a directory: {path}")
        
        matches = []
        try:
            for item in search_path.rglob(pattern):
                try:
                    relative_path = item.relative_to(self.root_path)
                    stat = item.stat()
                    matches.append({
                        "name": item.name,
                        "path": str(relative_path),
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": stat.st_mtime
                    })
                except (OSError, ValueError):
                    continue
        except Exception as e:
            raise ValueError(f"Search failed: {e}")
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(matches, indent=2)
            }]
        }
    
    async def _get_file_info(self, path: str) -> Dict[str, Any]:
        """Get information about a file or directory"""
        file_path = self._resolve_path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        stat = file_path.stat()
        info = {
            "name": file_path.name,
            "path": str(file_path.relative_to(self.root_path)),
            "type": "directory" if file_path.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "permissions": oct(stat.st_mode)[-3:],
            "absolute_path": str(file_path)
        }
        
        if file_path.is_dir():
            try:
                info["item_count"] = len(list(file_path.iterdir()))
            except PermissionError:
                info["item_count"] = "Permission denied"
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(info, indent=2)
            }]
        }


class FilesystemMCPProtocol:
    """MCP protocol handler for filesystem operations"""
    
    def __init__(self, server: FilesystemMCPServer):
        self.server = server
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "able-filesystem",
                "version": "1.0.0"
            }
        }
    
    async def handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP list tools request"""
        return {
            "tools": self.server.tools
        }
    
    async def handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP call tool request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            raise ValueError("Tool name is required")
        
        return await self.server.handle_tool_call(tool_name, arguments)


async def start_filesystem_server(root_path: Path) -> Optional[subprocess.Popen]:
    """Start the filesystem MCP server as a subprocess"""
    try:
        # Create the server script
        server_script = f'''
import asyncio
import json
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, "{Path(__file__).parent.parent}")

from mcp.filesystem_server import FilesystemMCPServer, FilesystemMCPProtocol

async def main():
    server = FilesystemMCPServer(Path("{root_path}"))
    protocol = FilesystemMCPProtocol(server)
    
    # Simple stdio-based MCP server
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
                
            request = json.loads(line.strip())
            method = request.get("method")
            params = request.get("params", {{}})
            request_id = request.get("id")
            
            if method == "initialize":
                response = await protocol.handle_initialize(params)
            elif method == "tools/list":
                response = await protocol.handle_list_tools(params)
            elif method == "tools/call":
                response = await protocol.handle_call_tool(params)
            else:
                response = {{"error": "Unknown method"}}
            
            result = {{
                "jsonrpc": "2.0",
                "id": request_id,
                "result": response
            }}
            
            print(json.dumps(result))
            sys.stdout.flush()
            
        except Exception as e:
            error_response = {{
                "jsonrpc": "2.0",
                "id": request.get("id", None),
                "error": {{"code": -1, "message": str(e)}}
            }}
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Write the script to a temporary file
        script_path = Path("/tmp/filesystem_mcp_server.py")
        script_path.write_text(server_script)
        
        # Start the subprocess
        process = subprocess.Popen([
            sys.executable, str(script_path)
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Started filesystem MCP server with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Failed to start filesystem MCP server: {e}")
        return None


if __name__ == "__main__":
    # For testing
    import asyncio
    
    async def test():
        root = Path("/tmp/test_filesystem")
        root.mkdir(exist_ok=True)
        
        server = FilesystemMCPServer(root)
        
        # Test file operations
        result = await server.handle_tool_call("write_file", {
            "path": "test.txt",
            "content": "Hello, MCP!"
        })
        print("Write result:", result)
        
        result = await server.handle_tool_call("read_file", {
            "path": "test.txt"
        })
        print("Read result:", result)
        
        result = await server.handle_tool_call("list_directory", {
            "path": ""
        })
        print("List result:", result)
    
    asyncio.run(test())