#!/usr/bin/env python3
"""
Create macOS Dock App for Able
Generates an app bundle that can be added to the dock and Applications folder
"""

import os
import sys
import shutil
from pathlib import Path

class AbleDockAppCreator:
    """Creates a macOS app bundle for Able"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.app_name = "Able"
        self.app_bundle_name = f"{self.app_name}.app"
        self.app_path = self.project_dir / self.app_bundle_name
        
    def create_app_structure(self):
        """Create the basic app bundle directory structure"""
        print(f"🏗️  Creating app bundle structure at {self.app_path}")
        
        # Remove existing app if it exists
        if self.app_path.exists():
            print(f"🗑️  Removing existing app bundle...")
            shutil.rmtree(self.app_path)
        
        # Create app bundle directories
        contents_dir = self.app_path / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        for directory in [contents_dir, macos_dir, resources_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory.relative_to(self.project_dir)}")
        
        return contents_dir, macos_dir, resources_dir
    
    def create_info_plist(self, contents_dir):
        """Create the Info.plist file for the app bundle"""
        print("📄 Creating Info.plist...")
        
        plist_data = {
            'CFBundleDisplayName': 'Able',
            'CFBundleName': 'Able',
            'CFBundleIdentifier': 'com.able.research-assistant',
            'CFBundleVersion': '3.0.0',
            'CFBundleShortVersionString': '3.0.0',
            'CFBundleExecutable': 'able_launcher',
            'CFBundleIconFile': 'able_icon.icns',
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': 'ABLE',
            'LSMinimumSystemVersion': '10.15',
            'NSHighResolutionCapable': True,
            'LSApplicationCategoryType': 'public.app-category.productivity',
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'PDF Document',
                    'CFBundleTypeRole': 'Editor',
                    'LSItemContentTypes': ['com.adobe.pdf'],
                    'CFBundleTypeExtensions': ['pdf']
                }
            ],
            'NSAppleEventsUsageDescription': 'Able needs access to system events to manage browser windows.',
            'NSDesktopFolderUsageDescription': 'Able may need access to desktop files for document processing.',
            'NSDocumentsFolderUsageDescription': 'Able needs access to documents for PDF processing.',
            'NSDownloadsFolderUsageDescription': 'Able may need access to downloads for PDF processing.'
        }
        
        plist_path = contents_dir / "Info.plist"
        
        # Write plist manually since we might not have the plist module
        with open(plist_path, 'w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDisplayName</key>
	<string>Able</string>
	<key>CFBundleName</key>
	<string>Able</string>
	<key>CFBundleIdentifier</key>
	<string>com.able.research-assistant</string>
	<key>CFBundleVersion</key>
	<string>3.0.0</string>
	<key>CFBundleShortVersionString</key>
	<string>3.0.0</string>
	<key>CFBundleExecutable</key>
	<string>able_launcher</string>
	<key>CFBundleIconFile</key>
	<string>able_icon</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleSignature</key>
	<string>ABLE</string>
	<key>LSMinimumSystemVersion</key>
	<string>10.15</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.productivity</string>
	<key>CFBundleDocumentTypes</key>
	<array>
		<dict>
			<key>CFBundleTypeName</key>
			<string>PDF Document</string>
			<key>CFBundleTypeRole</key>
			<string>Editor</string>
			<key>LSItemContentTypes</key>
			<array>
				<string>com.adobe.pdf</string>
			</array>
			<key>CFBundleTypeExtensions</key>
			<array>
				<string>pdf</string>
			</array>
		</dict>
	</array>
	<key>NSAppleEventsUsageDescription</key>
	<string>Able needs access to system events to manage browser windows.</string>
	<key>NSDesktopFolderUsageDescription</key>
	<string>Able may need access to desktop files for document processing.</string>
	<key>NSDocumentsFolderUsageDescription</key>
	<string>Able needs access to documents for PDF processing.</string>
	<key>NSDownloadsFolderUsageDescription</key>
	<string>Able may need access to downloads for PDF processing.</string>
</dict>
</plist>''')
        
        print(f"   Created: {plist_path.relative_to(self.project_dir)}")
    
    def create_launcher_script(self, macos_dir):
        """Create the executable launcher script"""
        print("🚀 Creating launcher script...")
        
        launcher_path = macos_dir / "able_launcher"
        
        launcher_content = f'''#!/bin/bash

# Able Enhanced macOS App Launcher

# The Able project directory (where the actual files are)
ABLE_PROJECT_DIR="{str(self.project_dir)}"

# Function to show error dialog
show_error() {{
    osascript -e "display alert \\"Able Error\\" message \\"$1\\" buttons {{\\"OK\\"}} default button \\"OK\\""
}}

# Function to show info notification
show_info() {{
    osascript -e "display notification \\"$1\\" with title \\"Able\\" sound name \\"Glass\\""
}}

# Change to the Able directory (with proper quoting)
if ! cd "$ABLE_PROJECT_DIR"; then
    show_error "Cannot access Able directory: $ABLE_PROJECT_DIR"
    exit 1
fi

# Check if we have the required files
if [[ ! -f "launch_able.py" ]]; then
    show_error "launch_able.py not found in: $ABLE_PROJECT_DIR\\n\\nPlease ensure Able is properly installed."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    show_error "Python 3 is required to run Able.\\n\\nPlease install Python 3 from python.org or use Homebrew:\\nbrew install python3"
    exit 1
fi

# Set up Node.js environment (nvm, homebrew, or system)
echo "Setting up Node.js environment..." >> "$LOG_FILE" 2>/dev/null || true

# Try to source nvm if it exists
if [[ -f "$HOME/.nvm/nvm.sh" ]]; then
    echo "Loading nvm..." >> "$LOG_FILE" 2>/dev/null || true
    export NVM_DIR="$HOME/.nvm"
    source "$NVM_DIR/nvm.sh" 2>/dev/null || true
    source "$NVM_DIR/bash_completion" 2>/dev/null || true
fi

# Add common Node.js paths to PATH
export PATH="$HOME/.nvm/versions/node/$(ls $HOME/.nvm/versions/node/ | tail -1)/bin:$PATH" 2>/dev/null || true
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PATH="/usr/local/lib/node_modules/npm/bin:$PATH"

# Check if npm is available
if ! command -v npm &> /dev/null; then
    show_error "Node.js and npm are required to run Able.\\n\\nPlease install Node.js from nodejs.org or use Homebrew:\\nbrew install node"
    exit 1
fi

echo "Node.js version: $(node --version 2>/dev/null || echo 'not found')" >> "$LOG_FILE" 2>/dev/null || true
echo "npm version: $(npm --version 2>/dev/null || echo 'not found')" >> "$LOG_FILE" 2>/dev/null || true

# Show startup notification
show_info "Starting Able..."

# Create log file for debugging
LOG_FILE="$ABLE_PROJECT_DIR/able_dock_launch.log"
echo "$(date): Starting Able from dock app" > "$LOG_FILE" 2>/dev/null || true
echo "PATH: $PATH" >> "$LOG_FILE" 2>/dev/null || true

# Launch Able with automatic port management
echo "Launching Able from dock..." >> "$LOG_FILE" 2>/dev/null || true
python3 launch_able.py >> "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

# Wait a bit and check if launch was successful
sleep 5
if ! kill -0 $LAUNCH_PID 2>/dev/null; then
    show_error "Failed to start Able. Check Console.app for details."
    exit 1
fi

# Wait a bit more for services to be ready
sleep 10

# Check if services are responding
if curl -s "http://localhost:3000" > /dev/null 2>&1; then
    # Services are ready, open browser
    echo "Opening browser..." >> "$LOG_FILE" 2>/dev/null || true
    open "http://localhost:3000"
    show_info "Able is ready! Browser opening..."
else
    # Services not ready yet, still open browser (it will load when ready)
    echo "Services still starting, opening browser anyway..." >> "$LOG_FILE" 2>/dev/null || true
    open "http://localhost:3000"
    show_info "Able is starting... Browser opened."
fi

# Keep the app "running" in the dock while Able is active
echo "Monitoring Able processes..." >> "$LOG_FILE" 2>/dev/null || true
while true; do
    # Check if backend or frontend processes are running
    if ! pgrep -f "main.py" > /dev/null 2>&1 && ! pgrep -f "react-scripts start" > /dev/null 2>&1; then
        echo "$(date): Able processes have stopped" >> "$LOG_FILE" 2>/dev/null || true
        break
    fi
    sleep 15
done

echo "$(date): Dock app exiting" >> "$LOG_FILE" 2>/dev/null || true
'''
        
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        # Make the launcher executable
        os.chmod(launcher_path, 0o755)
        print(f"   Created: {launcher_path.relative_to(self.project_dir)}")
    
    def convert_icon_to_icns(self, resources_dir):
        """Convert the PNG icon to ICNS format for macOS"""
        print("🎨 Processing app icon...")
        
        icon_png = self.project_dir / "Able Icon.png"
        icon_icns = resources_dir / "able_icon.icns"
        
        if not icon_png.exists():
            print(f"   ⚠️  Warning: Icon file not found at {icon_png}")
            print(f"   Using default icon instead")
            return False
        
        try:
            # Use sips (System Image Processing System) to convert PNG to ICNS
            # This is available on all macOS systems
            import subprocess
            
            # Create temporary iconset directory
            iconset_dir = self.project_dir / "able_icon.iconset"
            if iconset_dir.exists():
                shutil.rmtree(iconset_dir)
            iconset_dir.mkdir()
            
            # Generate different icon sizes
            sizes = [
                (16, "icon_16x16.png"),
                (32, "icon_16x16@2x.png"),
                (32, "icon_32x32.png"),
                (64, "icon_32x32@2x.png"),
                (128, "icon_128x128.png"),
                (256, "icon_128x128@2x.png"),
                (256, "icon_256x256.png"),
                (512, "icon_256x256@2x.png"),
                (512, "icon_512x512.png"),
                (1024, "icon_512x512@2x.png")
            ]
            
            for size, filename in sizes:
                output_path = iconset_dir / filename
                subprocess.run([
                    "sips", "-z", str(size), str(size), str(icon_png),
                    "--out", str(output_path)
                ], capture_output=True)
            
            # Convert iconset to icns
            subprocess.run([
                "iconutil", "-c", "icns", str(iconset_dir),
                "-o", str(icon_icns)
            ], capture_output=True)
            
            # Clean up iconset directory
            shutil.rmtree(iconset_dir)
            
            if icon_icns.exists():
                print(f"   ✅ Created: {icon_icns.relative_to(self.project_dir)}")
                return True
            else:
                print("   ⚠️  Failed to create ICNS file")
                return False
                
        except Exception as e:
            print(f"   ⚠️  Error creating icon: {e}")
            return False
    
    def create_app(self):
        """Create the complete app bundle"""
        print("🍎 Creating Able dock app...")
        print("=" * 50)
        
        # Create app structure
        contents_dir, macos_dir, resources_dir = self.create_app_structure()
        
        # Create Info.plist
        self.create_info_plist(contents_dir)
        
        # Create launcher script
        self.create_launcher_script(macos_dir)
        
        # Convert and add icon
        icon_success = self.convert_icon_to_icns(resources_dir)
        
        print("\n" + "=" * 50)
        if self.app_path.exists():
            print(f"✅ Successfully created {self.app_bundle_name}!")
            print(f"📁 Location: {self.app_path}")
            print("\n📋 Installation Instructions:")
            print("1. Double-click the app to test it")
            print("2. Drag it to your Applications folder")
            print("3. Right-click and select 'Add to Dock' if desired")
            print("4. Or drag from Applications to Dock")
            
            if not icon_success:
                print("\n⚠️  Icon conversion failed - app will use default icon")
            
            return True
        else:
            print("❌ Failed to create app bundle")
            return False
    
    def install_to_applications(self):
        """Install the app to the Applications folder"""
        applications_dir = Path("/Applications")
        target_path = applications_dir / self.app_bundle_name
        
        if not self.app_path.exists():
            print("❌ App bundle not found. Run create_app() first.")
            return False
        
        try:
            if target_path.exists():
                print(f"🗑️  Removing existing app from Applications...")
                shutil.rmtree(target_path)
            
            print(f"📦 Installing {self.app_bundle_name} to Applications...")
            shutil.copytree(self.app_path, target_path)
            
            print(f"✅ Successfully installed to {target_path}")
            print("🚀 You can now launch Able from Spotlight, Launchpad, or the Applications folder!")
            return True
            
        except PermissionError:
            print("❌ Permission denied. Try running with sudo or drag the app manually.")
            return False
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False

def main():
    """Main function to create the dock app"""
    creator = AbleDockAppCreator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        # Create and install to Applications
        if creator.create_app():
            creator.install_to_applications()
    else:
        # Just create the app bundle
        creator.create_app()

if __name__ == "__main__":
    main()