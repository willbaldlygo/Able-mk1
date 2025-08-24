"""
SQLite MCP Server for Able
Provides secure SQLite operations for metadata and custom databases
"""

import asyncio
import json
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

logger = logging.getLogger(__name__)

class SQLiteMCPServer:
    """MCP server for SQLite operations"""
    
    def __init__(self, allowed_dbs: List[Path], root_path: Path):
        self.allowed_dbs = [Path(db).resolve() for db in allowed_dbs]
        self.root_path = root_path.resolve()
        
        # Add default databases
        self._add_default_databases()
        
        self.tools = [
            {
                "name": "list_databases",
                "description": "List available databases",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "list_tables",
                "description": "List tables in a database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database path (relative to root or absolute allowed path)"
                        }
                    },
                    "required": ["database"]
                }
            },
            {
                "name": "describe_table",
                "description": "Get schema information for a table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database path"
                        },
                        "table": {
                            "type": "string",
                            "description": "Table name"
                        }
                    },
                    "required": ["database", "table"]
                }
            },
            {
                "name": "execute_query",
                "description": "Execute a SELECT query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database path"
                        },
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return",
                            "default": 100
                        }
                    },
                    "required": ["database", "query"]
                }
            },
            {
                "name": "get_document_metadata",
                "description": "Get metadata for documents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Specific document ID (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return",
                            "default": 50
                        }
                    }
                }
            },
            {
                "name": "search_chunks",
                "description": "Search document chunks by content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (text to search for in chunks)"
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Filter by specific document ID (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of chunks to return",
                            "default": 20
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_database_stats",
                "description": "Get statistics about a database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database path"
                        }
                    },
                    "required": ["database"]
                }
            }
        ]
    
    def _add_default_databases(self):
        """Add default databases that should be available"""
        # Look for ChromaDB database
        chroma_db_path = self.root_path / "data" / "vectordb"
        if chroma_db_path.exists():
            for db_file in chroma_db_path.glob("*.sqlite*"):
                if db_file not in self.allowed_dbs:
                    self.allowed_dbs.append(db_file)
        
        # Look for any other SQLite databases in data directory
        data_dir = self.root_path / "data"
        if data_dir.exists():
            for db_file in data_dir.rglob("*.db"):
                if db_file not in self.allowed_dbs:
                    self.allowed_dbs.append(db_file)
            for db_file in data_dir.rglob("*.sqlite"):
                if db_file not in self.allowed_dbs:
                    self.allowed_dbs.append(db_file)
            for db_file in data_dir.rglob("*.sqlite3"):
                if db_file not in self.allowed_dbs:
                    self.allowed_dbs.append(db_file)
    
    def _resolve_database_path(self, database: str) -> Path:
        """Resolve and validate database path"""
        # Try as absolute path first
        db_path = Path(database)
        if db_path.is_absolute():
            db_path = db_path.resolve()
            if db_path in self.allowed_dbs:
                return db_path
        
        # Try as relative to root path
        db_path = (self.root_path / database).resolve()
        if db_path in self.allowed_dbs:
            return db_path
        
        # Check if it's in allowed databases by name
        for allowed_db in self.allowed_dbs:
            if allowed_db.name == database or str(allowed_db).endswith(database):
                return allowed_db
        
        raise ValueError(f"Database {database} not found in allowed databases")
    
    def _validate_query(self, query: str) -> bool:
        """Validate that query is read-only"""
        query_upper = query.upper().strip()
        
        # Allow only SELECT statements and basic queries
        allowed_starts = ["SELECT", "WITH", "EXPLAIN", "PRAGMA"]
        
        if not any(query_upper.startswith(start) for start in allowed_starts):
            return False
        
        # Block dangerous keywords
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "REPLACE", "MERGE", "UPSERT"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False
        
        return True
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call and return result"""
        try:
            if tool_name == "list_databases":
                return await self._list_databases()
            elif tool_name == "list_tables":
                return await self._list_tables(arguments["database"])
            elif tool_name == "describe_table":
                return await self._describe_table(arguments["database"], arguments["table"])
            elif tool_name == "execute_query":
                return await self._execute_query(
                    arguments["database"],
                    arguments["query"],
                    arguments.get("limit", 100)
                )
            elif tool_name == "get_document_metadata":
                return await self._get_document_metadata(
                    arguments.get("document_id"),
                    arguments.get("limit", 50)
                )
            elif tool_name == "search_chunks":
                return await self._search_chunks(
                    arguments["query"],
                    arguments.get("document_id"),
                    arguments.get("limit", 20)
                )
            elif tool_name == "get_database_stats":
                return await self._get_database_stats(arguments["database"])
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
            }
    
    async def _list_databases(self) -> Dict[str, Any]:
        """List available databases"""
        databases = []
        
        for db_path in self.allowed_dbs:
            if db_path.exists():
                try:
                    stat = db_path.stat()
                    databases.append({
                        "name": db_path.name,
                        "path": str(db_path.relative_to(self.root_path)) if db_path.is_relative_to(self.root_path) else str(db_path),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "absolute_path": str(db_path)
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for database {db_path}: {e}")
                    continue
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(databases, indent=2)
            }]
        }
    
    async def _list_tables(self, database: str) -> Dict[str, Any]:
        """List tables in a database"""
        db_path = self._resolve_database_path(database)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database}")
        
        if AIOSQLITE_AVAILABLE:
            async with aiosqlite.connect(str(db_path)) as conn:
                cursor = await conn.execute(
                    "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
                )
                tables = await cursor.fetchall()
        else:
            # Fallback to synchronous sqlite3
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(
                    "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
                )
                tables = cursor.fetchall()
        
        table_info = []
        for name, table_type in tables:
            table_info.append({
                "name": name,
                "type": table_type
            })
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(table_info, indent=2)
            }]
        }
    
    async def _describe_table(self, database: str, table: str) -> Dict[str, Any]:
        """Get schema information for a table"""
        db_path = self._resolve_database_path(database)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database}")
        
        if AIOSQLITE_AVAILABLE:
            async with aiosqlite.connect(str(db_path)) as conn:
                # Get table schema
                cursor = await conn.execute(f"PRAGMA table_info({table})")
                columns = await cursor.fetchall()
                
                # Get table creation SQL
                cursor = await conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
                )
                creation_sql = await cursor.fetchone()
        else:
            # Fallback to synchronous sqlite3
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
                )
                creation_sql = cursor.fetchone()
        
        if not columns:
            raise ValueError(f"Table {table} not found")
        
        column_info = []
        for cid, name, data_type, notnull, default_value, pk in columns:
            column_info.append({
                "name": name,
                "type": data_type,
                "nullable": not bool(notnull),
                "default": default_value,
                "primary_key": bool(pk)
            })
        
        result = {
            "table": table,
            "columns": column_info
        }
        
        if creation_sql:
            result["creation_sql"] = creation_sql[0]
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        }
    
    async def _execute_query(self, database: str, query: str, limit: int) -> Dict[str, Any]:
        """Execute a SELECT query"""
        if not self._validate_query(query):
            raise ValueError("Only SELECT queries are allowed")
        
        db_path = self._resolve_database_path(database)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database}")
        
        # Add LIMIT if not present
        query_upper = query.upper().strip()
        if "LIMIT" not in query_upper:
            query += f" LIMIT {limit}"
        
        if AIOSQLITE_AVAILABLE:
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query)
                rows = await cursor.fetchall()
                
                # Get column names
                if rows:
                    columns = list(rows[0].keys())
                    data = [dict(row) for row in rows]
                else:
                    columns = []
                    data = []
        else:
            # Fallback to synchronous sqlite3
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                
                if rows:
                    columns = list(rows[0].keys())
                    data = [dict(row) for row in rows]
                else:
                    columns = []
                    data = []
        
        result = {
            "query": query,
            "columns": columns,
            "rows": data,
            "row_count": len(data)
        }
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        }
    
    async def _get_document_metadata(self, document_id: Optional[str], limit: int) -> Dict[str, Any]:
        """Get metadata for documents from Able's metadata system"""
        # Check for metadata.json file
        metadata_file = self.root_path / "data" / "metadata.json"
        
        if metadata_file.exists():
            # Use JSON metadata
            try:
                metadata_content = json.loads(metadata_file.read_text())
                
                if document_id:
                    # Filter by specific document
                    for doc in metadata_content.get("documents", []):
                        if doc.get("id") == document_id:
                            return {
                                "content": [{
                                    "type": "text",
                                    "text": json.dumps([doc], indent=2)
                                }]
                            }
                    raise ValueError(f"Document {document_id} not found")
                else:
                    # Return all documents (limited)
                    documents = metadata_content.get("documents", [])[:limit]
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps(documents, indent=2)
                        }]
                    }
            except Exception as e:
                logger.error(f"Error reading metadata.json: {e}")
        
        # Fallback: look for ChromaDB
        chroma_dbs = [db for db in self.allowed_dbs if "chroma" in str(db).lower()]
        
        if not chroma_dbs:
            return {
                "content": [{
                    "type": "text",
                    "text": "No document metadata found"
                }]
            }
        
        # Try to query ChromaDB
        try:
            db_path = chroma_dbs[0]
            query = """
                SELECT DISTINCT 
                    JSON_EXTRACT(metadata, '$.document_id') as document_id,
                    JSON_EXTRACT(metadata, '$.filename') as filename,
                    JSON_EXTRACT(metadata, '$.page_count') as page_count,
                    COUNT(*) as chunk_count
                FROM embeddings 
                GROUP BY document_id
                LIMIT ?
            """
            
            if document_id:
                query = """
                    SELECT 
                        JSON_EXTRACT(metadata, '$.document_id') as document_id,
                        JSON_EXTRACT(metadata, '$.filename') as filename,
                        JSON_EXTRACT(metadata, '$.page_count') as page_count,
                        COUNT(*) as chunk_count
                    FROM embeddings 
                    WHERE JSON_EXTRACT(metadata, '$.document_id') = ?
                    GROUP BY document_id
                """
                params = (document_id,)
            else:
                params = (limit,)
            
            if AIOSQLITE_AVAILABLE:
                async with aiosqlite.connect(str(db_path)) as conn:
                    cursor = await conn.execute(query, params)
                    rows = await cursor.fetchall()
            else:
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                documents.append({
                    "document_id": row[0],
                    "filename": row[1],
                    "page_count": row[2],
                    "chunk_count": row[3]
                })
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(documents, indent=2)
                }]
            }
        
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error accessing document metadata: {e}"
                }]
            }
    
    async def _search_chunks(self, query: str, document_id: Optional[str], limit: int) -> Dict[str, Any]:
        """Search document chunks by content"""
        chroma_dbs = [db for db in self.allowed_dbs if "chroma" in str(db).lower()]
        
        if not chroma_dbs:
            return {
                "content": [{
                    "type": "text",
                    "text": "No chunk database found"
                }]
            }
        
        try:
            db_path = chroma_dbs[0]
            
            # Build search query
            sql_query = """
                SELECT 
                    JSON_EXTRACT(metadata, '$.document_id') as document_id,
                    JSON_EXTRACT(metadata, '$.filename') as filename,
                    JSON_EXTRACT(metadata, '$.page_number') as page_number,
                    SUBSTR(document, 1, 200) as preview,
                    LENGTH(document) as chunk_length
                FROM embeddings 
                WHERE document LIKE ? COLLATE NOCASE
            """
            
            params = [f"%{query}%"]
            
            if document_id:
                sql_query += " AND JSON_EXTRACT(metadata, '$.document_id') = ?"
                params.append(document_id)
            
            sql_query += f" ORDER BY LENGTH(document) DESC LIMIT {limit}"
            
            if AIOSQLITE_AVAILABLE:
                async with aiosqlite.connect(str(db_path)) as conn:
                    cursor = await conn.execute(sql_query, params)
                    rows = await cursor.fetchall()
            else:
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute(sql_query, params)
                    rows = cursor.fetchall()
            
            chunks = []
            for row in rows:
                chunks.append({
                    "document_id": row[0],
                    "filename": row[1],
                    "page_number": row[2],
                    "preview": row[3] + "..." if len(row[3]) == 200 else row[3],
                    "chunk_length": row[4]
                })
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "query": query,
                        "total_matches": len(chunks),
                        "chunks": chunks
                    }, indent=2)
                }]
            }
        
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error searching chunks: {e}"
                }]
            }
    
    async def _get_database_stats(self, database: str) -> Dict[str, Any]:
        """Get statistics about a database"""
        db_path = self._resolve_database_path(database)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {database}")
        
        try:
            if AIOSQLITE_AVAILABLE:
                async with aiosqlite.connect(str(db_path)) as conn:
                    # Get table count
                    cursor = await conn.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    )
                    table_count = (await cursor.fetchone())[0]
                    
                    # Get database size info
                    cursor = await conn.execute("PRAGMA page_count")
                    page_count = (await cursor.fetchone())[0]
                    
                    cursor = await conn.execute("PRAGMA page_size")
                    page_size = (await cursor.fetchone())[0]
                    
                    # Get table sizes
                    cursor = await conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                    tables = await cursor.fetchall()
                    
                    table_stats = []
                    for (table_name,) in tables:
                        try:
                            cursor = await conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                            row_count = (await cursor.fetchone())[0]
                            table_stats.append({
                                "name": table_name,
                                "row_count": row_count
                            })
                        except Exception as e:
                            table_stats.append({
                                "name": table_name,
                                "row_count": f"Error: {e}"
                            })
            
            else:
                # Fallback to synchronous sqlite3
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                    )
                    table_count = cursor.fetchone()[0]
                    
                    cursor = conn.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]
                    
                    cursor = conn.execute("PRAGMA page_size")
                    page_size = cursor.fetchone()[0]
                    
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                    tables = cursor.fetchall()
                    
                    table_stats = []
                    for (table_name,) in tables:
                        try:
                            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                            row_count = cursor.fetchone()[0]
                            table_stats.append({
                                "name": table_name,
                                "row_count": row_count
                            })
                        except Exception as e:
                            table_stats.append({
                                "name": table_name,
                                "row_count": f"Error: {e}"
                            })
            
            file_stat = db_path.stat()
            
            stats = {
                "database": str(db_path.name),
                "file_size": file_stat.st_size,
                "file_size_mb": round(file_stat.st_size / 1024 / 1024, 2),
                "table_count": table_count,
                "page_count": page_count,
                "page_size": page_size,
                "calculated_size": page_count * page_size,
                "modified": file_stat.st_mtime,
                "tables": table_stats
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(stats, indent=2)
                }]
            }
        
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error getting database stats: {e}"
                }]
            }


class SQLiteMCPProtocol:
    """MCP protocol handler for SQLite operations"""
    
    def __init__(self, server: SQLiteMCPServer):
        self.server = server
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "able-sqlite",
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


async def start_sqlite_server(allowed_dbs: List[Path], root_path: Path) -> Optional[subprocess.Popen]:
    """Start the SQLite MCP server as a subprocess"""
    try:
        # Create the server script
        server_script = f'''
import asyncio
import json
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, "{Path(__file__).parent.parent}")

from mcp.sqlite_server import SQLiteMCPServer, SQLiteMCPProtocol

async def main():
    allowed_dbs = {[str(db) for db in allowed_dbs]}
    root_path = Path("{root_path}")
    server = SQLiteMCPServer([Path(db) for db in allowed_dbs], root_path)
    protocol = SQLiteMCPProtocol(server)
    
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
        script_path = Path("/tmp/sqlite_mcp_server.py")
        script_path.write_text(server_script)
        
        # Start the subprocess
        process = subprocess.Popen([
            sys.executable, str(script_path)
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Started SQLite MCP server with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Failed to start SQLite MCP server: {e}")
        return None


if __name__ == "__main__":
    # For testing
    import asyncio
    
    async def test():
        root = Path("/tmp/test_sqlite")
        root.mkdir(exist_ok=True)
        
        # Create a test database
        test_db = root / "test.db"
        
        if AIOSQLITE_AVAILABLE:
            async with aiosqlite.connect(str(test_db)) as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT,
                        content TEXT
                    )
                """)
                await conn.execute("""
                    INSERT INTO documents VALUES ('1', 'test.txt', 'Hello, world!')
                """)
                await conn.commit()
        else:
            with sqlite3.connect(str(test_db)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT,
                        content TEXT
                    )
                """)
                conn.execute("""
                    INSERT INTO documents VALUES ('1', 'test.txt', 'Hello, world!')
                """)
                conn.commit()
        
        server = SQLiteMCPServer([test_db], root)
        
        # Test operations
        result = await server.handle_tool_call("list_databases", {})
        print("Databases:", result)
        
        result = await server.handle_tool_call("list_tables", {"database": "test.db"})
        print("Tables:", result)
        
        result = await server.handle_tool_call("execute_query", {
            "database": "test.db",
            "query": "SELECT * FROM documents"
        })
        print("Query result:", result)
    
    asyncio.run(test())