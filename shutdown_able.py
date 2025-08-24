#!/usr/bin/env python3
"""
Comprehensive Able Service Shutdown Script
Cleanly terminates all Able-related services and logs session information.
"""

import os
import signal
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

def find_able_processes():
    """Find all Able-related processes."""
    processes = {}
    
    try:
        # Get all processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        
        for line in lines:
            parts = line.split(None, 10)  # Split into max 11 parts
            if len(parts) >= 11:
                user, pid, *_, command = parts
                pid = int(pid)
                
                # Check for Able-related processes
                command_lower = command.lower()
                if any(keyword in command_lower for keyword in [
                    'able_launcher', 'launch_able.py', 'main.py',
                    '/able3_main_withvoicemode/'
                ]) and 'grep' not in command_lower:
                    
                    # Determine process type
                    if 'able_launcher' in command:
                        processes['launcher'] = {'pid': pid, 'command': command}
                    elif 'launch_able.py' in command:
                        processes['python_launcher'] = {'pid': pid, 'command': command}
                    elif 'main.py' in command and 'backend' in command:
                        processes['backend'] = {'pid': pid, 'command': command}
                    elif 'python3 -m http.server 3001' in command or 'port.*3001' in command:
                        processes['frontend'] = {'pid': pid, 'command': command}
                    else:
                        processes[f'other_{pid}'] = {'pid': pid, 'command': command}
                        
    except Exception as e:
        print(f"Error finding processes: {e}")
    
    return processes

def check_ports():
    """Check what's running on Able's ports."""
    ports_status = {}
    
    for port in [3001, 8000, 11434]:
        try:
            result = subprocess.run(['lsof', '-i', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines:
                    parts = lines[0].split()
                    command = parts[0] if parts else 'unknown'
                    pid = parts[1] if len(parts) > 1 else 'unknown'
                    ports_status[port] = {'pid': pid, 'command': command}
                else:
                    ports_status[port] = None
            else:
                ports_status[port] = None
        except Exception:
            ports_status[port] = None
            
    return ports_status

def create_session_log(processes, ports_status):
    """Create session shutdown log."""
    try:
        # Get document count
        metadata_file = Path("data/metadata.json")
        doc_count = 0
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                doc_count = len(metadata)
        
        # Create log entry
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "action": "shutdown",
            "documents_processed": doc_count,
            "processes_terminated": len(processes),
            "ports_cleared": len([p for p in ports_status.values() if p is not None]),
            "process_details": {k: v['pid'] for k, v in processes.items()},
            "port_status": ports_status
        }
        
        # Append to session log
        log_file = Path("session_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(session_info) + "\n")
            
        return session_info
        
    except Exception as e:
        print(f"Error creating session log: {e}")
        return {"error": str(e)}

def terminate_process(pid, name, timeout=5):
    """Terminate a process gracefully, then forcefully if needed."""
    try:
        print(f"🔄 Terminating {name} (PID: {pid})...")
        
        # Try graceful termination first
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to terminate
        for _ in range(timeout * 2):  # Check every 0.5 seconds
            try:
                os.kill(pid, 0)  # Check if process still exists
                time.sleep(0.5)
            except ProcessLookupError:
                print(f"✅ {name} terminated gracefully")
                return True
                
        # If still running, force kill
        print(f"⚠️  {name} didn't terminate gracefully, force killing...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        print(f"✅ {name} force terminated")
        return True
        
    except ProcessLookupError:
        print(f"ℹ️  {name} was already terminated")
        return True
    except PermissionError:
        print(f"❌ Permission denied to terminate {name}")
        return False
    except Exception as e:
        print(f"❌ Error terminating {name}: {e}")
        return False

def main():
    """Main shutdown procedure."""
    print("🔴 Able Comprehensive Shutdown")
    print("=" * 40)
    
    # Find all processes
    print("🔍 Finding Able processes...")
    processes = find_able_processes()
    
    if not processes:
        print("ℹ️  No Able processes found running")
    else:
        print(f"📊 Found {len(processes)} Able processes:")
        for name, info in processes.items():
            print(f"  • {name}: PID {info['pid']}")
    
    # Check ports
    print("\n🔍 Checking ports...")
    ports_status = check_ports()
    for port, status in ports_status.items():
        if status:
            print(f"  • Port {port}: {status['command']} (PID: {status['pid']})")
        else:
            print(f"  • Port {port}: free")
    
    # Create session log
    print("\n📝 Creating session log...")
    session_info = create_session_log(processes, ports_status)
    
    # Terminate processes in order
    if processes:
        print("\n🛑 Terminating processes...")
        
        # Terminate in reverse dependency order
        termination_order = ['backend', 'frontend', 'python_launcher', 'launcher']
        
        for process_type in termination_order:
            if process_type in processes:
                success = terminate_process(
                    processes[process_type]['pid'], 
                    process_type
                )
                if success:
                    del processes[process_type]
        
        # Terminate any remaining processes
        for name, info in processes.items():
            terminate_process(info['pid'], name)
    
    # Final port check
    print("\n🔍 Final port check...")
    final_ports = check_ports()
    active_ports = [port for port, status in final_ports.items() if status is not None]
    
    if active_ports:
        print(f"⚠️  Ports still active: {active_ports}")
    else:
        print("✅ All ports cleared")
    
    print(f"\n🎉 Shutdown complete!")
    print(f"📊 Session summary:")
    print(f"   • Documents processed: {session_info.get('documents_processed', 'unknown')}")
    print(f"   • Processes terminated: {session_info.get('processes_terminated', 0)}")
    print(f"   • Session logged: session_log.jsonl")
    print("\n💡 To start Able again, run: python3 launch_able.py")

if __name__ == "__main__":
    main()