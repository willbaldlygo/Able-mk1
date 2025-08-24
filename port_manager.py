#!/usr/bin/env python3
"""
Port Management Utility for Able
Checks required ports and kills conflicting processes before startup
"""

import subprocess
import sys
import time
import signal
import os
from typing import List, Dict, Optional, Tuple

class PortManager:
    """Manages port availability for Able startup"""
    
    def __init__(self):
        self.required_ports = {
            3001: "New UI Frontend Server",
            8000: "Backend FastAPI Server", 
            11434: "Ollama Local AI Server (optional)"
        }
        
    def check_port_availability(self, port: int) -> Tuple[bool, Optional[int]]:
        """
        Check if a port is available
        Returns: (is_available, process_id_if_occupied)
        """
        try:
            # Use lsof to check what's using the port
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Port is occupied, return the PID
                pids = result.stdout.strip().split('\n')
                return False, int(pids[0]) if pids[0] else None
            else:
                # Port is available
                return True, None
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            # If lsof fails, assume port is available
            return True, None
    
    def get_process_info(self, pid: int) -> Dict[str, str]:
        """Get information about a process"""
        try:
            # Get process command
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'comm=,args='],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(None, 1)
                    command = parts[0] if parts else "unknown"
                    args = parts[1] if len(parts) > 1 else ""
                    return {"command": command, "args": args}
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
            
        return {"command": "unknown", "args": ""}
    
    def kill_process_on_port(self, port: int, force: bool = False) -> bool:
        """
        Kill process occupying a port
        Returns True if successful
        """
        is_available, pid = self.check_port_availability(port)
        
        if is_available:
            print(f"✓ Port {port} is already available")
            return True
            
        if not pid:
            print(f"✗ Port {port} appears occupied but no PID found")
            return False
            
        process_info = self.get_process_info(pid)
        print(f"! Port {port} is occupied by PID {pid} ({process_info['command']})")
        
        try:
            # First try graceful termination
            if not force:
                print(f"  Attempting graceful termination of PID {pid}...")
                os.kill(pid, signal.SIGTERM)
                
                # Wait up to 5 seconds for graceful shutdown
                for _ in range(10):
                    time.sleep(0.5)
                    is_available, _ = self.check_port_availability(port)
                    if is_available:
                        print(f"✓ Process {pid} terminated gracefully")
                        return True
                
                print(f"  Graceful termination failed, forcing...")
            
            # Force kill if graceful failed or force requested
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            
            # Verify the process is gone
            is_available, _ = self.check_port_availability(port)
            if is_available:
                print(f"✓ Process {pid} terminated forcefully")
                return True
            else:
                print(f"✗ Failed to terminate process {pid}")
                return False
                
        except ProcessLookupError:
            # Process already gone
            print(f"✓ Process {pid} was already terminated")
            return True
        except PermissionError:
            print(f"✗ Permission denied to kill process {pid}")
            return False
        except Exception as e:
            print(f"✗ Error killing process {pid}: {e}")
            return False
    
    def prepare_ports(self, force_kill: bool = False, skip_optional: bool = True) -> bool:
        """
        Prepare all required ports for Able startup
        Returns True if all required ports are ready
        """
        print("🔍 Checking port availability for Able...")
        print("=" * 50)
        
        all_ready = True
        
        for port, description in self.required_ports.items():
            # Skip optional ports if requested
            if skip_optional and port == 11434:
                is_available, pid = self.check_port_availability(port)
                if is_available:
                    print(f"✓ Port {port} ({description}) - Available (optional)")
                else:
                    print(f"! Port {port} ({description}) - Occupied but skipping (optional)")
                continue
            
            print(f"\nChecking port {port} ({description}):")
            
            is_available, pid = self.check_port_availability(port)
            
            if is_available:
                print(f"✓ Port {port} is available")
            else:
                success = self.kill_process_on_port(port, force_kill)
                if not success:
                    all_ready = False
                    print(f"✗ Failed to free port {port}")
                else:
                    print(f"✓ Port {port} is now available")
        
        print("\n" + "=" * 50)
        if all_ready:
            print("✅ All required ports are ready for Able!")
        else:
            print("❌ Some required ports could not be freed")
            
        return all_ready
    
    def show_port_status(self):
        """Display current status of all required ports"""
        print("📊 Current port status:")
        print("=" * 50)
        
        for port, description in self.required_ports.items():
            is_available, pid = self.check_port_availability(port)
            
            status = "Available" if is_available else f"Occupied (PID: {pid})"
            icon = "✅" if is_available else "🔴"
            
            print(f"{icon} Port {port:>5} ({description}): {status}")
            
            if not is_available and pid:
                process_info = self.get_process_info(pid)
                print(f"     Process: {process_info['command']} {process_info['args'][:50]}...")


def main():
    """Command line interface for port management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Able Port Management Utility")
    parser.add_argument("--check", action="store_true", help="Check port status only")
    parser.add_argument("--prepare", action="store_true", help="Prepare ports for startup")
    parser.add_argument("--force", action="store_true", help="Force kill processes immediately")
    parser.add_argument("--include-optional", action="store_true", help="Include optional ports (like Ollama)")
    parser.add_argument("--port", type=int, help="Manage specific port only")
    
    args = parser.parse_args()
    
    manager = PortManager()
    
    if args.check:
        manager.show_port_status()
    elif args.prepare:
        skip_optional = not args.include_optional
        success = manager.prepare_ports(force_kill=args.force, skip_optional=skip_optional)
        sys.exit(0 if success else 1)
    elif args.port:
        success = manager.kill_process_on_port(args.port, force=args.force)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()