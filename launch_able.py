#!/usr/bin/env python3
"""
Quick Launch Script for Able
Simple wrapper that uses port management and starts the system
"""

import sys
import os
from pathlib import Path

# Add current directory to path to import port_manager
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from port_manager import PortManager
from start_able import AbleStartup

def main():
    """Simple main launcher with port management"""
    print("🚀 Quick Launch - Able PDF Research Assistant")
    print("=" * 50)
    
    # Quick port check and cleanup
    port_manager = PortManager()
    print("🔍 Checking and preparing ports...")
    
    if not port_manager.prepare_ports(force_kill=False, skip_optional=True):
        print("❌ Could not prepare required ports. Try running with --force-kill:")
        print("   python3 launch_able.py --force")
        return False
    
    print("✅ Ports ready, starting Able...")
    print("-" * 50)
    
    # Start Able
    startup = AbleStartup()
    return startup.start(skip_port_check=True)  # Skip port check since we just did it

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        # Force kill processes on ports first
        PortManager().prepare_ports(force_kill=True, skip_optional=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check port status
        PortManager().show_port_status()
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python3 launch_able.py [OPTIONS]")
        print("Options:")
        print("  --force    Force kill processes on required ports before starting")
        print("  --check    Only check port status")
        print("  --help     Show this help")
        sys.exit(0)
    
    success = main()
    sys.exit(0 if success else 1)