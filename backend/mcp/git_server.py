"""
Git MCP Server for Able
Provides Git operations with confirmation system for destructive actions
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import git
    from git import Repo, InvalidGitRepositoryError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    git = None
    Repo = None
    InvalidGitRepositoryError = Exception

logger = logging.getLogger(__name__)

class GitMCPServer:
    """MCP server for Git operations"""
    
    def __init__(self, allowed_repos: List[Path]):
        self.allowed_repos = [Path(repo).resolve() for repo in allowed_repos]
        self.tools = [
            {
                "name": "git_status",
                "description": "Get the status of a Git repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository (optional, defaults to first allowed repo)"
                        }
                    }
                }
            },
            {
                "name": "git_log",
                "description": "Get commit history",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of commits to retrieve",
                            "default": 10
                        },
                        "branch": {
                            "type": "string",
                            "description": "Branch to get log for (defaults to current branch)"
                        }
                    }
                }
            },
            {
                "name": "git_diff",
                "description": "Show differences between commits, branches, or working directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "commit1": {
                            "type": "string",
                            "description": "First commit/branch to compare (optional)"
                        },
                        "commit2": {
                            "type": "string",
                            "description": "Second commit/branch to compare (optional)"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Specific file to diff (optional)"
                        }
                    }
                }
            },
            {
                "name": "git_branch",
                "description": "List, create, or switch branches",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "action": {
                            "type": "string",
                            "enum": ["list", "create", "switch", "delete"],
                            "description": "Action to perform"
                        },
                        "branch_name": {
                            "type": "string",
                            "description": "Branch name (for create, switch, delete actions)"
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "git_add",
                "description": "Stage changes for commit",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files to stage (use '.' for all)"
                        }
                    },
                    "required": ["files"]
                }
            },
            {
                "name": "git_commit",
                "description": "Create a commit",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "message": {
                            "type": "string",
                            "description": "Commit message"
                        },
                        "author_name": {
                            "type": "string",
                            "description": "Author name (optional)"
                        },
                        "author_email": {
                            "type": "string",
                            "description": "Author email (optional)"
                        }
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "git_push",
                "description": "Push changes to remote repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "remote": {
                            "type": "string",
                            "description": "Remote name (defaults to 'origin')",
                            "default": "origin"
                        },
                        "branch": {
                            "type": "string",
                            "description": "Branch to push (defaults to current branch)"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirm the push operation",
                            "default": false
                        }
                    }
                }
            },
            {
                "name": "git_pull",
                "description": "Pull changes from remote repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "remote": {
                            "type": "string",
                            "description": "Remote name (defaults to 'origin')",
                            "default": "origin"
                        },
                        "branch": {
                            "type": "string",
                            "description": "Branch to pull (defaults to current branch)"
                        }
                    }
                }
            },
            {
                "name": "git_clone",
                "description": "Clone a repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Repository URL to clone"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination path (relative to allowed repos)"
                        },
                        "branch": {
                            "type": "string",
                            "description": "Specific branch to clone (optional)"
                        }
                    },
                    "required": ["url", "destination"]
                }
            },
            {
                "name": "git_remote",
                "description": "Manage remotes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to the Git repository"
                        },
                        "action": {
                            "type": "string",
                            "enum": ["list", "add", "remove", "set-url"],
                            "description": "Action to perform"
                        },
                        "name": {
                            "type": "string",
                            "description": "Remote name"
                        },
                        "url": {
                            "type": "string",
                            "description": "Remote URL (for add/set-url)"
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "discover_repos",
                "description": "Discover Git repositories in allowed paths",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def _get_repo_path(self, repo_path: Optional[str] = None) -> Path:
        """Resolve and validate repository path"""
        if not repo_path and self.allowed_repos:
            # Use first allowed repo as default
            return self.allowed_repos[0]
        
        if not repo_path:
            raise ValueError("No repository path provided and no default available")
        
        # Convert to absolute path
        path = Path(repo_path).resolve()
        
        # Check if path is within allowed repositories
        for allowed_repo in self.allowed_repos:
            try:
                path.relative_to(allowed_repo)
                return path
            except ValueError:
                continue
        
        raise ValueError(f"Repository path {path} not within allowed repositories")
    
    def _get_repo(self, repo_path: Optional[str] = None) -> Repo:
        """Get Git repository object"""
        if not GIT_AVAILABLE:
            raise RuntimeError("GitPython is not available")
        
        path = self._get_repo_path(repo_path)
        
        try:
            return Repo(path)
        except InvalidGitRepositoryError:
            raise ValueError(f"Path {path} is not a Git repository")
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call and return result"""
        try:
            if tool_name == "git_status":
                return await self._git_status(arguments.get("repo_path"))
            elif tool_name == "git_log":
                return await self._git_log(
                    arguments.get("repo_path"),
                    arguments.get("limit", 10),
                    arguments.get("branch")
                )
            elif tool_name == "git_diff":
                return await self._git_diff(
                    arguments.get("repo_path"),
                    arguments.get("commit1"),
                    arguments.get("commit2"),
                    arguments.get("file_path")
                )
            elif tool_name == "git_branch":
                return await self._git_branch(
                    arguments.get("repo_path"),
                    arguments["action"],
                    arguments.get("branch_name")
                )
            elif tool_name == "git_add":
                return await self._git_add(
                    arguments.get("repo_path"),
                    arguments["files"]
                )
            elif tool_name == "git_commit":
                return await self._git_commit(
                    arguments.get("repo_path"),
                    arguments["message"],
                    arguments.get("author_name"),
                    arguments.get("author_email")
                )
            elif tool_name == "git_push":
                return await self._git_push(
                    arguments.get("repo_path"),
                    arguments.get("remote", "origin"),
                    arguments.get("branch"),
                    arguments.get("confirm", False)
                )
            elif tool_name == "git_pull":
                return await self._git_pull(
                    arguments.get("repo_path"),
                    arguments.get("remote", "origin"),
                    arguments.get("branch")
                )
            elif tool_name == "git_clone":
                return await self._git_clone(
                    arguments["url"],
                    arguments["destination"],
                    arguments.get("branch")
                )
            elif tool_name == "git_remote":
                return await self._git_remote(
                    arguments.get("repo_path"),
                    arguments["action"],
                    arguments.get("name"),
                    arguments.get("url")
                )
            elif tool_name == "discover_repos":
                return await self._discover_repos()
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {str(e)}"}]
            }
    
    async def _git_status(self, repo_path: Optional[str]) -> Dict[str, Any]:
        """Get repository status"""
        repo = self._get_repo(repo_path)
        
        status_info = {
            "branch": repo.active_branch.name,
            "staged": [item.a_path for item in repo.index.diff("HEAD")],
            "unstaged": [item.a_path for item in repo.index.diff(None)],
            "untracked": repo.untracked_files,
            "ahead": 0,
            "behind": 0
        }
        
        # Check if we're ahead/behind remote
        try:
            remote = repo.remote()
            remote_branch = f"{remote.name}/{repo.active_branch.name}"
            if remote_branch in [ref.name for ref in repo.refs]:
                commits_ahead = list(repo.iter_commits(f"{remote_branch}..HEAD"))
                commits_behind = list(repo.iter_commits(f"HEAD..{remote_branch}"))
                status_info["ahead"] = len(commits_ahead)
                status_info["behind"] = len(commits_behind)
        except:
            pass  # No remote or other issues
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(status_info, indent=2)
            }]
        }
    
    async def _git_log(self, repo_path: Optional[str], limit: int, branch: Optional[str]) -> Dict[str, Any]:
        """Get commit history"""
        repo = self._get_repo(repo_path)
        
        if branch:
            commits = list(repo.iter_commits(branch, max_count=limit))
        else:
            commits = list(repo.iter_commits(max_count=limit))
        
        log_info = []
        for commit in commits:
            log_info.append({
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:7],
                "author": f"{commit.author.name} <{commit.author.email}>",
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
                "files_changed": len(commit.stats.files)
            })
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(log_info, indent=2)
            }]
        }
    
    async def _git_diff(self, repo_path: Optional[str], commit1: Optional[str], commit2: Optional[str], file_path: Optional[str]) -> Dict[str, Any]:
        """Show differences"""
        repo = self._get_repo(repo_path)
        
        if commit1 and commit2:
            # Diff between two commits
            diff = repo.git.diff(commit1, commit2, file_path or "")
        elif commit1:
            # Diff between commit and working directory
            diff = repo.git.diff(commit1, file_path or "")
        else:
            # Diff between working directory and index
            diff = repo.git.diff(file_path or "")
        
        return {
            "content": [{
                "type": "text",
                "text": diff if diff else "No differences found"
            }]
        }
    
    async def _git_branch(self, repo_path: Optional[str], action: str, branch_name: Optional[str]) -> Dict[str, Any]:
        """Branch operations"""
        repo = self._get_repo(repo_path)
        
        if action == "list":
            branches = []
            for branch in repo.branches:
                is_current = branch == repo.active_branch
                branches.append({
                    "name": branch.name,
                    "current": is_current,
                    "commit": branch.commit.hexsha[:7]
                })
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(branches, indent=2)
                }]
            }
        
        elif action == "create":
            if not branch_name:
                raise ValueError("Branch name required for create action")
            
            new_branch = repo.create_head(branch_name)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Created branch: {branch_name}"
                }]
            }
        
        elif action == "switch":
            if not branch_name:
                raise ValueError("Branch name required for switch action")
            
            repo.heads[branch_name].checkout()
            return {
                "content": [{
                    "type": "text",
                    "text": f"Switched to branch: {branch_name}"
                }]
            }
        
        elif action == "delete":
            if not branch_name:
                raise ValueError("Branch name required for delete action")
            
            # Safety check: don't delete current branch
            if repo.active_branch.name == branch_name:
                raise ValueError("Cannot delete currently active branch")
            
            repo.delete_head(branch_name)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Deleted branch: {branch_name}"
                }]
            }
    
    async def _git_add(self, repo_path: Optional[str], files: List[str]) -> Dict[str, Any]:
        """Stage files"""
        repo = self._get_repo(repo_path)
        
        if "." in files:
            repo.git.add(".")
            return {
                "content": [{
                    "type": "text",
                    "text": "Staged all changes"
                }]
            }
        else:
            repo.index.add(files)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Staged files: {', '.join(files)}"
                }]
            }
    
    async def _git_commit(self, repo_path: Optional[str], message: str, author_name: Optional[str], author_email: Optional[str]) -> Dict[str, Any]:
        """Create commit"""
        repo = self._get_repo(repo_path)
        
        # Check if there are staged changes
        if not repo.index.diff("HEAD"):
            return {
                "content": [{
                    "type": "text",
                    "text": "No staged changes to commit"
                }]
            }
        
        commit_kwargs = {"message": message}
        
        if author_name and author_email:
            from git import Actor
            commit_kwargs["author"] = Actor(author_name, author_email)
        
        commit = repo.index.commit(**commit_kwargs)
        
        return {
            "content": [{
                "type": "text",
                "text": f"Created commit {commit.hexsha[:7]}: {message}"
            }]
        }
    
    async def _git_push(self, repo_path: Optional[str], remote: str, branch: Optional[str], confirm: bool) -> Dict[str, Any]:
        """Push changes"""
        if not confirm:
            return {
                "content": [{
                    "type": "text",
                    "text": "Push operation requires confirmation. Set 'confirm': true to proceed."
                }]
            }
        
        repo = self._get_repo(repo_path)
        
        remote_obj = repo.remote(remote)
        
        if branch:
            push_info = remote_obj.push(f"refs/heads/{branch}")
        else:
            push_info = remote_obj.push()
        
        results = []
        for info in push_info:
            results.append({
                "local_ref": str(info.local_ref),
                "remote_ref": str(info.remote_ref),
                "flags": info.flags
            })
        
        return {
            "content": [{
                "type": "text",
                "text": f"Push completed:\n{json.dumps(results, indent=2)}"
            }]
        }
    
    async def _git_pull(self, repo_path: Optional[str], remote: str, branch: Optional[str]) -> Dict[str, Any]:
        """Pull changes"""
        repo = self._get_repo(repo_path)
        
        remote_obj = repo.remote(remote)
        
        if branch:
            pull_info = remote_obj.pull(f"refs/heads/{branch}")
        else:
            pull_info = remote_obj.pull()
        
        results = []
        for info in pull_info:
            results.append({
                "ref": str(info.ref),
                "flags": info.flags,
                "commit": info.commit.hexsha[:7] if info.commit else None
            })
        
        return {
            "content": [{
                "type": "text",
                "text": f"Pull completed:\n{json.dumps(results, indent=2)}"
            }]
        }
    
    async def _git_clone(self, url: str, destination: str, branch: Optional[str]) -> Dict[str, Any]:
        """Clone repository"""
        # Ensure destination is within allowed paths
        dest_path = None
        for allowed_repo in self.allowed_repos:
            try:
                potential_path = (allowed_repo / destination).resolve()
                potential_path.relative_to(allowed_repo)
                dest_path = potential_path
                break
            except ValueError:
                continue
        
        if not dest_path:
            raise ValueError(f"Destination {destination} not within allowed repositories")
        
        clone_kwargs = {"to_path": str(dest_path)}
        if branch:
            clone_kwargs["branch"] = branch
        
        repo = Repo.clone_from(url, **clone_kwargs)
        
        return {
            "content": [{
                "type": "text",
                "text": f"Cloned repository from {url} to {dest_path}"
            }]
        }
    
    async def _git_remote(self, repo_path: Optional[str], action: str, name: Optional[str], url: Optional[str]) -> Dict[str, Any]:
        """Manage remotes"""
        repo = self._get_repo(repo_path)
        
        if action == "list":
            remotes = []
            for remote in repo.remotes:
                remotes.append({
                    "name": remote.name,
                    "urls": list(remote.urls)
                })
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(remotes, indent=2)
                }]
            }
        
        elif action == "add":
            if not name or not url:
                raise ValueError("Remote name and URL required for add action")
            
            remote = repo.create_remote(name, url)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Added remote {name}: {url}"
                }]
            }
        
        elif action == "remove":
            if not name:
                raise ValueError("Remote name required for remove action")
            
            repo.delete_remote(name)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Removed remote: {name}"
                }]
            }
        
        elif action == "set-url":
            if not name or not url:
                raise ValueError("Remote name and URL required for set-url action")
            
            remote = repo.remote(name)
            remote.set_url(url)
            return {
                "content": [{
                    "type": "text",
                    "text": f"Updated remote {name} URL to: {url}"
                }]
            }
    
    async def _discover_repos(self) -> Dict[str, Any]:
        """Discover Git repositories in allowed paths"""
        repos = []
        
        for allowed_path in self.allowed_repos:
            if not allowed_path.exists():
                continue
            
            # Check if the path itself is a repo
            try:
                repo = Repo(allowed_path)
                repos.append({
                    "path": str(allowed_path),
                    "branch": repo.active_branch.name if repo.active_branch else "No branch",
                    "remotes": [remote.name for remote in repo.remotes]
                })
            except InvalidGitRepositoryError:
                pass
            
            # Search for repos in subdirectories
            for item in allowed_path.rglob(".git"):
                if item.is_dir():
                    repo_path = item.parent
                    try:
                        repo = Repo(repo_path)
                        repos.append({
                            "path": str(repo_path),
                            "branch": repo.active_branch.name if repo.active_branch else "No branch",
                            "remotes": [remote.name for remote in repo.remotes]
                        })
                    except InvalidGitRepositoryError:
                        continue
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(repos, indent=2)
            }]
        }


class GitMCPProtocol:
    """MCP protocol handler for Git operations"""
    
    def __init__(self, server: GitMCPServer):
        self.server = server
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "able-git",
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


async def start_git_server(allowed_repos: List[Path]) -> Optional[subprocess.Popen]:
    """Start the Git MCP server as a subprocess"""
    if not GIT_AVAILABLE:
        logger.error("GitPython not available, cannot start Git MCP server")
        return None
    
    try:
        # Create the server script
        server_script = f'''
import asyncio
import json
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, "{Path(__file__).parent.parent}")

from mcp.git_server import GitMCPServer, GitMCPProtocol

async def main():
    allowed_repos = {[str(repo) for repo in allowed_repos]}
    server = GitMCPServer([Path(repo) for repo in allowed_repos])
    protocol = GitMCPProtocol(server)
    
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
        script_path = Path("/tmp/git_mcp_server.py")
        script_path.write_text(server_script)
        
        # Start the subprocess
        process = subprocess.Popen([
            sys.executable, str(script_path)
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"Started Git MCP server with PID {process.pid}")
        return process
        
    except Exception as e:
        logger.error(f"Failed to start Git MCP server: {e}")
        return None


if __name__ == "__main__":
    # For testing
    import asyncio
    
    async def test():
        allowed_repos = [Path("/tmp/test_repo")]
        allowed_repos[0].mkdir(exist_ok=True)
        
        # Initialize a test repo
        if GIT_AVAILABLE:
            try:
                repo = Repo.init(allowed_repos[0])
                test_file = allowed_repos[0] / "test.txt"
                test_file.write_text("Hello, Git!")
                repo.index.add([str(test_file)])
                repo.index.commit("Initial commit")
                
                server = GitMCPServer(allowed_repos)
                
                # Test status
                result = await server.handle_tool_call("git_status", {})
                print("Status result:", result)
                
                # Test log
                result = await server.handle_tool_call("git_log", {"limit": 5})
                print("Log result:", result)
                
            except Exception as e:
                print(f"Test error: {e}")
        else:
            print("GitPython not available for testing")
    
    asyncio.run(test())