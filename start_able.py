#!/usr/bin/env python3
"""
Able Startup Script
Manages ports and launches both frontend and backend services
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path
from port_manager import PortManager

class AbleStartup:
    """Manages the complete Able startup process"""
    
    def __init__(self):
        self.port_manager = PortManager()
        self.processes = {}
        self.base_dir = Path(__file__).parent
        self.frontend_dir = self.base_dir / "new-ui"
        self.backend_dir = self.base_dir / "backend"
        self.running = False
        
    def check_dependencies(self) -> bool:
        """Check if all required directories and files exist"""
        print("🔍 Checking Able dependencies...")
        
        checks = [
            (self.frontend_dir.exists(), f"New UI directory: {self.frontend_dir}"),
            (self.backend_dir.exists(), f"Backend directory: {self.backend_dir}"),
            ((self.frontend_dir / "index.html").exists(), "New UI index.html"),
            ((self.backend_dir / "main.py").exists(), "Backend main.py"),
            ((self.backend_dir / "requirements.txt").exists(), "Backend requirements.txt"),
        ]
        
        all_good = True
        for check_passed, description in checks:
            status = "✅" if check_passed else "❌"
            print(f"{status} {description}")
            if not check_passed:
                all_good = False
        
        if not all_good:
            print("\n❌ Some dependencies are missing. Please check your Able installation.")
            
        return all_good
    
    def check_python_environment(self) -> bool:
        """Check if Python virtual environment is set up"""
        venv_path = self.backend_dir / "venv"
        
        if not venv_path.exists():
            print("⚠️  Python virtual environment not found. Creating...")
            try:
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], check=True, cwd=self.backend_dir)
                print("✅ Virtual environment created")
            except subprocess.CalledProcessError:
                print("❌ Failed to create virtual environment")
                return False
        
        # Check if requirements are installed
        pip_path = venv_path / "bin" / "pip"
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip.exe"
        
        try:
            # Install/update requirements
            print("📦 Installing/updating Python dependencies...")
            subprocess.run([
                str(pip_path), "install", "-r", "requirements.txt", "--quiet"
            ], check=True, cwd=self.backend_dir)
            print("✅ Python dependencies ready")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Python dependencies")
            return False
    
    def check_frontend_files(self) -> bool:
        """Check if new-ui files are present"""
        required_files = ["index.html", "script.js", "styles.css"]
        
        for file_name in required_files:
            file_path = self.frontend_dir / file_name
            if not file_path.exists():
                print(f"❌ Missing required new-ui file: {file_name}")
                return False
        
        print("✅ All new-ui files present")
        return True
    
    def start_backend(self) -> bool:
        """Start the backend FastAPI server"""
        print("🚀 Starting Able backend...")
        
        python_path = self.backend_dir / "venv" / "bin" / "python"
        if sys.platform == "win32":
            python_path = self.backend_dir / "venv" / "Scripts" / "python.exe"
        
        try:
            process = subprocess.Popen([
                str(python_path), "main.py"
            ], cwd=self.backend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['backend'] = process
            
            # Wait a moment and check if process started successfully
            time.sleep(3)
            if process.poll() is None:
                print("✅ Backend started successfully on port 8000")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Backend failed to start:")
                if stderr:
                    print(f"Error: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """Start the new UI frontend server"""
        print("🚀 Starting Able new UI...")
        
        try:
            new_ui_dir = self.base_dir / "new-ui"
            if not new_ui_dir.exists():
                print(f"❌ New UI directory not found: {new_ui_dir}")
                return False
            
            # Start Python HTTP server for the new UI on port 3001
            process = subprocess.Popen([
                "python3", "-m", "http.server", "3001"
            ], cwd=new_ui_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['frontend'] = process
            
            # Wait a bit for server to start
            print("⏳ Starting new UI server...")
            time.sleep(3)
            
            if process.poll() is None:
                print("✅ New UI started successfully on port 3001")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ New UI failed to start:")
                if stderr:
                    print(f"Error: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start new UI: {e}")
            return False
    
    def check_services_health(self) -> bool:
        """Check if both services are responding"""
        print("🏥 Checking service health...")
        
        # Check backend health
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend health check passed")
                backend_healthy = True
            else:
                print(f"⚠️  Backend health check failed: {response.status_code}")
                backend_healthy = False
        except ImportError:
            print("⚠️  Backend health check skipped (requests module not available)")
            backend_healthy = True  # Assume healthy if we can't check
        except Exception as e:
            print(f"❌ Backend health check failed: {e}")
            backend_healthy = False
        
        # Check new UI availability
        try:
            import requests
            response = requests.get("http://localhost:3001", timeout=5)
            if response.status_code == 200:
                print("✅ New UI health check passed")
                frontend_healthy = True
            else:
                print(f"⚠️  New UI health check failed: {response.status_code}")
                frontend_healthy = False
        except ImportError:
            print("⚠️  New UI health check skipped (requests module not available)")
            frontend_healthy = True  # Assume healthy if we can't check
        except Exception as e:
            print(f"❌ New UI health check failed: {e}")
            frontend_healthy = False
        
        return backend_healthy and frontend_healthy
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            
            for service_name, process in self.processes.items():
                if process.poll() is not None:
                    print(f"⚠️  {service_name} process has stopped unexpectedly")
                    # Could implement restart logic here
            
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down Able...")
        self.stop()
    
    def stop(self):
        """Stop all Able services"""
        self.running = False
        
        print("🛑 Stopping Able services...")
        
        for service_name, process in self.processes.items():
            try:
                print(f"  Stopping {service_name}...")
                process.terminate()
                
                # Wait up to 10 seconds for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f"✅ {service_name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {service_name}...")
                    process.kill()
                    process.wait()
                    print(f"✅ {service_name} force stopped")
                    
            except Exception as e:
                print(f"❌ Error stopping {service_name}: {e}")
        
        print("✅ All services stopped")
        sys.exit(0)
    
    def start(self, skip_port_check=False, force_kill=False):
        """Start the complete Able system"""
        print("🚀 Starting Able - PDF Research Assistant")
        print("=" * 50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Prepare ports
        if not skip_port_check:
            if not self.port_manager.prepare_ports(force_kill=force_kill):
                print("❌ Cannot start Able - ports are not available")
                return False
        
        # Check environments
        if not self.check_python_environment():
            return False
            
        if not self.check_frontend_files():
            return False
        
        # Start services
        if not self.start_backend():
            return False
        
        if not self.start_frontend():
            self.stop()
            return False
        
        # Final health check
        time.sleep(5)  # Give services time to fully start
        if not self.check_services_health():
            print("⚠️  Some services may not be fully ready")
        
        self.running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        print("\n" + "=" * 50)
        print("✅ Able is now running!")
        print("🌐 New UI: http://localhost:3001")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop Able")
        print("=" * 50)
        
        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        
        return True


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Able PDF Research Assistant")
    parser.add_argument("--skip-port-check", action="store_true", 
                       help="Skip automatic port management")
    parser.add_argument("--force-kill", action="store_true",
                       help="Force kill processes on required ports")
    parser.add_argument("--check-ports", action="store_true",
                       help="Only check port status")
    
    args = parser.parse_args()
    
    if args.check_ports:
        PortManager().show_port_status()
        return
    
    startup = AbleStartup()
    success = startup.start(
        skip_port_check=args.skip_port_check,
        force_kill=args.force_kill
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()