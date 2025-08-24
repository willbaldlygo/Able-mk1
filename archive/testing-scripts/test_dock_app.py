#!/usr/bin/env python3
"""
Test the dock app functionality
"""

import subprocess
import time
import requests
from pathlib import Path

def test_dock_app():
    """Test the dock app launcher"""
    print("🧪 Testing Able Dock App")
    print("=" * 30)
    
    project_dir = Path(__file__).parent
    launcher_path = project_dir / "Able.app" / "Contents" / "MacOS" / "able_launcher"
    
    if not launcher_path.exists():
        print("❌ Launcher not found. Run ./install_dock_app.sh first")
        return False
    
    print("✅ Launcher found")
    print(f"📍 Testing: {launcher_path}")
    print()
    
    # Start the launcher
    print("🚀 Starting launcher...")
    process = subprocess.Popen([str(launcher_path)], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    # Wait for startup
    print("⏳ Waiting for startup (20 seconds)...")
    time.sleep(20)
    
    # Check if services are responding
    print("🔍 Checking services...")
    
    try:
        # Check backend
        backend_response = requests.get("http://localhost:8000/health", timeout=5)
        if backend_response.status_code == 200:
            print("✅ Backend is responding")
            backend_ok = True
        else:
            print(f"⚠️  Backend returned status: {backend_response.status_code}")
            backend_ok = False
    except Exception as e:
        print(f"❌ Backend not responding: {e}")
        backend_ok = False
    
    try:
        # Check frontend
        frontend_response = requests.get("http://localhost:3000", timeout=5)
        if frontend_response.status_code == 200:
            print("✅ Frontend is responding")
            frontend_ok = True
        else:
            print(f"⚠️  Frontend returned status: {frontend_response.status_code}")
            frontend_ok = False
    except Exception as e:
        print(f"❌ Frontend not responding: {e}")
        frontend_ok = False
    
    # Check log file
    log_file = project_dir / "able_dock_launch.log"
    if log_file.exists():
        print("✅ Log file created")
        print("📄 Recent log entries:")
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines[-5:]:  # Show last 5 lines
                print(f"   {line.strip()}")
    else:
        print("⚠️  No log file found")
    
    print("\n" + "=" * 30)
    if backend_ok and frontend_ok:
        print("🎉 SUCCESS: Dock app is working!")
        print("🌐 Able is available at: http://localhost:3000")
        return True
    else:
        print("❌ FAILED: Some services are not responding")
        return False

if __name__ == "__main__":
    test_dock_app()