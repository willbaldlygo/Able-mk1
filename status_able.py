#!/usr/bin/env python3
"""
Able Service Status Checker
Shows current status of all Able services and provides management options.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

def check_ports():
    """Check what's running on Able's ports."""
    ports_status = {}
    port_info = {
        3001: {"name": "Frontend (New UI)", "service": "HTTP Server"},
        8000: {"name": "Backend", "service": "FastAPI Server"},
        11434: {"name": "Ollama", "service": "Local AI Server"}
    }
    
    for port, info in port_info.items():
        try:
            result = subprocess.run(['lsof', '-i', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines:
                    parts = lines[0].split()
                    command = parts[0] if parts else 'unknown'
                    pid = parts[1] if len(parts) > 1 else 'unknown'
                    ports_status[port] = {
                        'status': 'active',
                        'pid': pid, 
                        'command': command,
                        'name': info['name'],
                        'service': info['service']
                    }
                else:
                    ports_status[port] = {
                        'status': 'inactive',
                        'name': info['name'],
                        'service': info['service']
                    }
            else:
                ports_status[port] = {
                    'status': 'inactive',
                    'name': info['name'],
                    'service': info['service']
                }
        except Exception:
            ports_status[port] = {
                'status': 'error',
                'name': info['name'],
                'service': info['service']
            }
            
    return ports_status

def check_api_health():
    """Check if backend API is responding."""
    if not HAS_REQUESTS:
        return {'status': 'unknown', 'error': 'requests module not available'}
        
    try:
        response = requests.get('http://localhost:8000/health', timeout=3)
        if response.status_code == 200:
            data = response.json()
            return {
                'status': 'healthy',
                'documents_count': data.get('documents_count', 0),
                'vector_chunks': data.get('vector_stats', {}).get('total_documents', 0)
            }
        else:
            return {'status': 'unhealthy', 'code': response.status_code}
    except Exception as e:
        # Use curl as fallback
        try:
            result = subprocess.run(['curl', '-s', 'http://localhost:8000/health'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return {'status': 'reachable', 'method': 'curl'}
            else:
                return {'status': 'unreachable', 'error': 'Connection refused'}
        except Exception:
            return {'status': 'unreachable', 'error': 'Connection refused'}

def get_document_count():
    """Get document count from metadata."""
    try:
        metadata_file = Path("data/metadata.json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                return len(metadata)
        return 0
    except Exception:
        return "unknown"

def get_last_session():
    """Get info about last session from log."""
    try:
        log_file = Path("session_log.jsonl")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return {
                        'timestamp': last_entry.get('timestamp', 'unknown'),
                        'action': last_entry.get('action', 'unknown'),
                        'documents': last_entry.get('documents_processed', 0)
                    }
        return None
    except Exception:
        return None

def print_status():
    """Print comprehensive status report."""
    print("📊 Able Service Status")
    print("=" * 50)
    print(f"⏰ Status checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Port status
    print("\n🔌 Port Status:")
    ports_status = check_ports()
    all_active = True
    
    for port, status in ports_status.items():
        icon = "✅" if status['status'] == 'active' else "❌"
        name = status['name']
        service = status['service']
        
        if status['status'] == 'active':
            pid = status.get('pid', 'unknown')
            print(f"  {icon} Port {port}: {name} ({service}) - PID {pid}")
        else:
            print(f"  {icon} Port {port}: {name} ({service}) - {status['status']}")
            all_active = False
    
    # API Health
    print("\n🏥 API Health:")
    if ports_status[8000]['status'] == 'active':
        health = check_api_health()
        if health['status'] == 'healthy':
            print(f"  ✅ Backend API: Healthy")
            print(f"     📄 Documents: {health.get('documents_count', 0)}")
            print(f"     🧠 Vector chunks: {health.get('vector_chunks', 0)}")
        else:
            print(f"  ❌ Backend API: {health['status']} - {health.get('error', 'Unknown error')}")
    else:
        print(f"  ❌ Backend API: Not running")
    
    # Data Status
    print("\n📁 Data Status:")
    doc_count = get_document_count()
    print(f"  📄 Documents in metadata: {doc_count}")
    
    # Check key directories
    data_dirs = [
        ("sources", "PDF storage"),
        ("data/vectordb", "Vector database"),
        ("data/graphrag", "GraphRAG data"),
    ]
    
    for dir_path, description in data_dirs:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.rglob('*'))) if path.is_dir() else 1
                print(f"  ✅ {description}: {file_count} files")
            else:
                print(f"  ✅ {description}: exists")
        else:
            print(f"  ❌ {description}: missing")
    
    # Last session info
    print("\n📝 Last Session:")
    last_session = get_last_session()
    if last_session:
        timestamp = last_session['timestamp']
        action = last_session['action']
        docs = last_session['documents']
        print(f"  🕐 {timestamp}")
        print(f"  🎬 Action: {action}")
        print(f"  📄 Documents: {docs}")
    else:
        print("  ℹ️  No session log found")
    
    # Overall status
    print("\n🎯 Overall Status:")
    if all_active:
        print("  ✅ All services running normally")
        print("  🌐 Access UI at: http://localhost:3001")
    else:
        print("  ⚠️  Some services not running")
        print("  💡 To start: python3 launch_able.py")
        print("  🛑 To shutdown: python3 shutdown_able.py")

def main():
    """Main status checking function."""
    print_status()

if __name__ == "__main__":
    main()