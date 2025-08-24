#!/usr/bin/env python3
"""
Create Simple Dock App with hardcoded paths
This version uses hardcoded paths to avoid environment issues
"""

import os
import sys
import shutil
from pathlib import Path

def create_simple_dock_app():
    """Create a simple dock app with hardcoded paths"""
    
    current_dir = Path(__file__).parent
    
    # Detect npm path
    npm_path = None
    possible_npm_paths = [
        "/Users/will/.nvm/versions/node/v24.4.0/bin/npm",
        "/opt/homebrew/bin/npm", 
        "/usr/local/bin/npm",
        "/usr/bin/npm"
    ]
    
    for path in possible_npm_paths:
        if Path(path).exists():
            npm_path = path
            break
    
    if not npm_path:
        print("❌ Could not find npm. Please ensure Node.js is installed.")
        return False
    
    node_dir = Path(npm_path).parent
    print(f"✅ Found Node.js at: {node_dir}")
    
    # Create app bundle
    app_name = "Able"
    app_bundle_name = f"{app_name}.app"
    app_path = current_dir / app_bundle_name
    
    print(f"🏗️  Creating simple app bundle: {app_path}")
    
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
    
    # Create simple launcher script with hardcoded paths
    launcher_path = macos_dir / "able_launcher"
    launcher_content = f'''#!/bin/bash

# Able Simple macOS App Launcher with hardcoded paths

# Project directory
ABLE_PROJECT_DIR="{str(current_dir)}"

# Node.js paths (hardcoded for reliability)
NODE_DIR="{str(node_dir)}"
export PATH="$NODE_DIR:$PATH"

# Function to show error dialog
show_error() {{
    osascript -e "display alert \\"Able Error\\" message \\"$1\\" buttons {{\\"OK\\"}} default button \\"OK\\""
}}

# Function to show notification
show_info() {{
    osascript -e "display notification \\"$1\\" with title \\"Able\\" sound name \\"Glass\\""
}}

# Change to project directory
cd "$ABLE_PROJECT_DIR" || {{
    show_error "Cannot access Able directory: $ABLE_PROJECT_DIR"
    exit 1
}}

# Check required files
if [[ ! -f "launch_able.py" ]]; then
    show_error "launch_able.py not found in: $ABLE_PROJECT_DIR"
    exit 1
fi

# Check Python 3
if ! command -v python3 &> /dev/null; then
    show_error "Python 3 is required. Install from python.org or: brew install python3"
    exit 1
fi

# Check Node.js/npm with hardcoded path
if [[ ! -f "$NODE_DIR/npm" ]]; then
    show_error "npm not found at: $NODE_DIR/npm\\n\\nNode.js may need to be reinstalled."
    exit 1
fi

# Show startup notification
show_info "Starting Able..."

# Create log file
LOG_FILE="$ABLE_PROJECT_DIR/able_simple_dock.log"
echo "$(date): Starting Able with simple dock app" > "$LOG_FILE"
echo "Node.js path: $NODE_DIR" >> "$LOG_FILE"
echo "npm path: $NODE_DIR/npm" >> "$LOG_FILE"
echo "npm version: $($NODE_DIR/npm --version 2>/dev/null)" >> "$LOG_FILE"

# Launch Able
echo "Launching Able..." >> "$LOG_FILE"
python3 launch_able.py >> "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!

# Wait and check
sleep 5
if ! kill -0 $LAUNCH_PID 2>/dev/null; then
    show_error "Failed to start Able. Check able_simple_dock.log for details."
    exit 1
fi

# Wait for services
sleep 15

# Open browser
if curl -s "http://localhost:3001" > /dev/null 2>&1; then
    echo "Opening browser - services ready" >> "$LOG_FILE"
    open "http://localhost:3001"
    show_info "Able is ready! Browser opening..."
else
    echo "Opening browser - services still starting" >> "$LOG_FILE"  
    open "http://localhost:3001"
    show_info "Able starting... Browser opened."
fi

# Monitor processes
echo "Monitoring processes..." >> "$LOG_FILE"
while true; do
    if ! pgrep -f "main.py" > /dev/null 2>&1 && ! pgrep -f "http.server 3001" > /dev/null 2>&1; then
        echo "$(date): Processes stopped" >> "$LOG_FILE"
        break
    fi
    sleep 15
done

echo "$(date): Dock app exiting" >> "$LOG_FILE"
'''
    
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    # Make executable
    os.chmod(launcher_path, 0o755)
    
    # Copy icon if available
    icon_png = current_dir / "Able Icon.png"
    if icon_png.exists():
        try:
            import subprocess
            
            # Create iconset
            iconset_dir = current_dir / "able_icon.iconset"
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
            
            # Convert to icns
            icon_icns = resources_dir / "able_icon.icns"
            subprocess.run([
                "iconutil", "-c", "icns", str(iconset_dir),
                "-o", str(icon_icns)
            ], capture_output=True)
            
            # Clean up
            shutil.rmtree(iconset_dir)
            
        except Exception as e:
            print(f"⚠️  Icon creation failed: {e}")
    
    print("✅ Simple dock app created!")
    print(f"📁 Location: {app_path}")
    return True

if __name__ == "__main__":
    create_simple_dock_app()