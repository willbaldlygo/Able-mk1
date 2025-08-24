#!/usr/bin/env python3
"""
Test script to demonstrate port management functionality
"""

import time
import subprocess
import sys
from port_manager import PortManager

def create_test_server(port):
    """Start a simple HTTP server on a port for testing"""
    print(f"🧪 Starting test server on port {port}...")
    return subprocess.Popen([
        'python3', '-m', 'http.server', str(port)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    """Demonstrate port management functionality"""
    print("🧪 Port Management System Test")
    print("=" * 40)
    
    manager = PortManager()
    
    # Show initial status
    print("1️⃣  Initial port status:")
    manager.show_port_status()
    
    # Start test servers on required ports
    print("\n2️⃣  Starting test servers...")
    test_servers = []
    
    try:
        # Start servers on ports 3000 and 8000
        server_3000 = create_test_server(3000)
        server_8000 = create_test_server(8000)
        test_servers.extend([server_3000, server_8000])
        
        time.sleep(2)  # Give servers time to start
        
        print("3️⃣  Port status with test servers running:")
        manager.show_port_status()
        
        print("\n4️⃣  Testing port management (prepare_ports)...")
        success = manager.prepare_ports(force_kill=False, skip_optional=True)
        
        if success:
            print("✅ Port management successful!")
        else:
            print("❌ Port management failed!")
        
        print("\n5️⃣  Final port status:")
        manager.show_port_status()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    finally:
        # Clean up test servers
        print("\n🧹 Cleaning up test servers...")
        for server in test_servers:
            try:
                server.terminate()
                server.wait(timeout=5)
            except:
                try:
                    server.kill()
                except:
                    pass
        
        print("✅ Test complete!")

if __name__ == "__main__":
    main()