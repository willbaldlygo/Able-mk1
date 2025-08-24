#!/bin/bash

# Install Able to macOS Applications and Dock
# This script creates the app bundle and installs it

set -e

echo "🍎 Able macOS Installation Script"
echo "=================================="

# Create the app bundle
echo "🏗️  Creating Able.app bundle..."
python3 create_dock_app.py

if [[ ! -d "Able.app" ]]; then
    echo "❌ Failed to create Able.app"
    exit 1
fi

# Install to Applications folder
echo ""
echo "📦 Installing to Applications folder..."

if [[ -d "/Applications/Able.app" ]]; then
    echo "🗑️  Removing existing installation..."
    rm -rf "/Applications/Able.app"
fi

cp -R "Able.app" "/Applications/"

if [[ -d "/Applications/Able.app" ]]; then
    echo "✅ Successfully installed to /Applications/Able.app"
else
    echo "❌ Installation failed - you may need administrator permissions"
    echo "💡 Try: sudo ./install_dock_app.sh"
    echo "    or drag Able.app to Applications manually"
    exit 1
fi

echo ""
echo "🚀 Installation Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Able is now installed in Applications"
echo "🔍 You can find it in:"
echo "   • Spotlight search (⌘+Space, type 'Able')"
echo "   • Launchpad"
echo "   • Applications folder"
echo ""
echo "🎯 To add to Dock:"
echo "   • Right-click Able in Applications → Add to Dock"
echo "   • Or drag Able from Applications to your Dock"
echo ""
echo "🔧 To uninstall later:"
echo "   • Delete /Applications/Able.app"
echo ""
echo "Happy researching! 📚✨"