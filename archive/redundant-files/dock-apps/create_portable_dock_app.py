#!/usr/bin/env python3
"""
Create Portable macOS Dock App for Able
This version prompts the user for the Able project location to make it more flexible
"""

import os
import sys
import shutil
from pathlib import Path

def create_portable_app():
    """Create a portable version that asks for project location"""
    
    print("🍎 Creating Portable Able Dock App")
    print("=" * 40)
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Let user specify or confirm the Able project directory
    print(f"📍 Current detected location: {current_dir}")
    response = input("📁 Is this the correct Able project directory? (y/n): ").lower().strip()
    
    if response != 'y':
        project_path = input("📂 Enter the full path to your Able project directory: ").strip()
        project_dir = Path(project_path).expanduser().resolve()
        
        if not project_dir.exists():
            print(f"❌ Directory not found: {project_dir}")
            return False
            
        if not (project_dir / "launch_able.py").exists():
            print(f"❌ launch_able.py not found in: {project_dir}")
            return False
    else:
        project_dir = current_dir
    
    print(f"✅ Using project directory: {project_dir}")
    
    # Create app bundle
    app_name = "Able"
    app_bundle_name = f"{app_name}.app"
    app_path = project_dir / app_bundle_name
    
    print(f"🏗️  Creating app bundle: {app_path}")
    
    # Remove existing app if it exists
    if app_path.exists():
        shutil.rmtree(app_path)
    
    # Create app bundle directories
    contents_dir = app_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    for directory in [contents_dir, macos_dir, resources_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create Info.plist
    plist_path = contents_dir / "Info.plist"
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
</dict>
</plist>''')
    
    # Create improved launcher script
    launcher_path = macos_dir / "able_launcher"
    launcher_content = f'''#!/bin/bash

# Able Portable macOS App Launcher

# Project directory (properly quoted for spaces)
ABLE_PROJECT_DIR="{str(project_dir)}"

# Function to show error dialog
show_error() {{
    osascript -e "display alert \\"Able Error\\" message \\"$1\\" buttons {{\\"OK\\"}} default button \\"OK\\""
}}

# Function to show info dialog
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

# Show startup notification
show_info "Starting Able..."

# Create log file for debugging
LOG_FILE="$ABLE_PROJECT_DIR/able_dock_launch.log"
echo "$(date): Starting Able from dock app" >> "$LOG_FILE"

# Launch Able with automatic port management
echo "Launching Able from dock..." >> "$LOG_FILE"
python3 launch_able.py >> "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

# Wait a bit and check if launch was successful
sleep 5
if ! kill -0 $LAUNCH_PID 2>/dev/null; then
    show_error "Failed to start Able. Check the log file at: $LOG_FILE"
    exit 1
fi

# Wait a bit more for services to be ready
sleep 10

# Check if services are responding
if curl -s "http://localhost:3000" > /dev/null 2>&1; then
    # Services are ready, open browser
    echo "Opening browser..." >> "$LOG_FILE"
    open "http://localhost:3000"
    show_info "Able is ready! Browser opening..."
else
    # Services not ready yet, still open browser (it will load when ready)
    echo "Services still starting, opening browser anyway..." >> "$LOG_FILE"
    open "http://localhost:3000"
    show_info "Able is starting... Browser opened."
fi

# Keep the app "running" in the dock while Able is active
echo "Monitoring Able processes..." >> "$LOG_FILE"
while true; do
    # Check if backend or frontend processes are running
    if ! pgrep -f "main.py" > /dev/null 2>&1 && ! pgrep -f "react-scripts start" > /dev/null 2>&1; then
        echo "$(date): Able processes have stopped" >> "$LOG_FILE"
        break
    fi
    sleep 15
done

echo "$(date): Dock app exiting" >> "$LOG_FILE"
'''
    
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    # Make the launcher executable
    os.chmod(launcher_path, 0o755)
    
    # Copy icon if available
    icon_png = project_dir / "Able Icon.png"
    if icon_png.exists():
        try:
            import subprocess
            
            # Create iconset
            iconset_dir = project_dir / "able_icon.iconset"
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
            icon_icns = resources_dir / "able_icon.icns"
            subprocess.run([
                "iconutil", "-c", "icns", str(iconset_dir),
                "-o", str(icon_icns)
            ], capture_output=True)
            
            # Clean up iconset directory
            shutil.rmtree(iconset_dir)
            
            if icon_icns.exists():
                print("✅ Icon created successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create icon: {e}")
    
    print("\n" + "=" * 40)
    print(f"✅ Portable Able.app created!")
    print(f"📁 Location: {app_path}")
    print("\n📋 Next Steps:")
    print("1. Test: Double-click the app to test it")
    print("2. Install: Drag it to your Applications folder")
    print("3. Dock: Right-click in Applications and 'Add to Dock'")
    print("\n💡 Tip: This app will always launch Able from:")
    print(f"   {project_dir}")
    
    return True

if __name__ == "__main__":
    create_portable_app()