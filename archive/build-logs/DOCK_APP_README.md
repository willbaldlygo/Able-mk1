# 🍎 Able macOS Dock App

This creates a native macOS application that you can launch from the dock, Applications folder, Spotlight, or Launchpad.

## 🚀 Quick Installation

```bash
./install_dock_app.sh
```

This will:
1. ✅ Create the Able.app bundle with custom icon
2. 📦 Install it to your Applications folder  
3. 🎯 Make it available in Spotlight and Launchpad

## 🎮 Manual Installation

```bash
# Just create the app bundle
python3 create_dock_app.py

# Create and install to Applications
python3 create_dock_app.py --install
```

## 🎯 Adding to Dock

After installation:
1. Open **Applications** folder in Finder
2. Find **Able** app
3. **Right-click** → **Add to Dock**
4. Or simply **drag** Able to your Dock

## ⚡ What the App Does

When you launch Able from the dock:

1. 🔍 **Port Management** - Automatically checks and frees ports 3000 and 8000
2. 🚀 **Backend Startup** - Launches the FastAPI server  
3. 🌐 **Frontend Startup** - Launches the React development server
4. 🌍 **Browser Launch** - Opens http://localhost:3000 automatically
5. 👀 **Process Monitoring** - Keeps running while Able is active

## 📁 App Bundle Structure

```
Able.app/
├── Contents/
│   ├── Info.plist          # App configuration
│   ├── MacOS/
│   │   └── able_launcher   # Launch script
│   └── Resources/
│       └── able_icon.icns  # Custom icon
```

## 🛠️ Technical Details

- **Bundle ID**: `com.able.research-assistant`
- **Version**: 3.0.0  
- **Minimum macOS**: 10.15 (Catalina)
- **Icon**: Converted from "Able Icon.png" to ICNS format
- **Launch Method**: Bash script wrapper around Python launcher

## 🗑️ Uninstallation

Simply delete the app:
```bash
rm -rf "/Applications/Able.app"
```

Or drag it to Trash from Applications folder.

## 🔧 Troubleshooting

### "launch_able.py not found" Error
**Fixed in latest version!** The app now uses hardcoded paths to the project directory.

If you still get this error:
- The app was created before the fix - recreate it: `./install_dock_app.sh`
- Or use the portable version: `python3 create_portable_dock_app.py`

### App Won't Launch
- Check that Python 3 is installed: `python3 --version`
- Verify launch_able.py exists in the project directory  
- Check `able_dock_launch.log` in the project directory for errors
- Look in Console app for system error messages

### Services Don't Start
- Check if ports 3000/8000 are free: `python3 port_manager.py --check`
- Try manual launch first: `python3 launch_able.py`
- Review the log file: `able_dock_launch.log`

### Missing Icon
- Ensure "Able Icon.png" exists in the project root
- Icon is automatically converted to ICNS format

### Permission Issues  
- Run installation with: `sudo ./install_dock_app.sh`
- Or manually drag Able.app to Applications

### Path with Spaces Issues
- The current fix handles spaces in paths properly
- If issues persist, use the portable version

## 🎨 Customization

To modify the app:
1. Edit `create_dock_app.py` 
2. Update `Info.plist` settings
3. Replace icon file and regenerate
4. Reinstall: `./install_dock_app.sh`

---

🚀 **Enjoy your native Able experience!** The app provides a seamless way to launch Able directly from macOS without terminal commands.