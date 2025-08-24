# Able mk I - New Web UI

A modern web interface for the Able PDF Research Assistant, built following the ABLE_WEBUI_INTEGRATION_GUIDE.md specifications.

## Features

- **Modern Design**: Matches the provided UI mockup with gradient header and rounded corners
- **Document Management**: Drag & drop PDF upload with library view
- **Real-time Chat**: Interactive chat interface with source attribution
- **Voice Input**: Speech-to-text capabilities for hands-free operation
- **System Status**: Live backend health monitoring
- **Responsive Layout**: Clean two-panel design

## Quick Start

1. **Ensure Able backend is running**:
   ```bash
   cd "/Users/will/AVI BUILD/Able3_Main_WithVoiceMode"
   ./start_able.sh
   ```

2. **Open the new UI**:
   ```bash
   cd new-ui
   python3 -m http.server 3001
   ```

3. **Access the application**:
   - Open http://localhost:3001 in your browser
   - Backend API: http://localhost:8000

## File Structure

```
new-ui/
├── index.html      # Main HTML structure
├── styles.css      # CSS styling matching the design
├── script.js       # JavaScript with full API integration
└── README.md       # This file
```

## API Integration

Fully implements the AVI_WEBUI_INTEGRATION_GUIDE.md specifications:

- ✅ Document upload (POST /upload)
- ✅ Document listing (GET /documents)  
- ✅ Document deletion (DELETE /documents/{id})
- ✅ Chat queries (POST /chat)
- ✅ Health monitoring (GET /health)
- ✅ Error handling and user feedback
- ✅ Source attribution display
- ✅ Voice input support

## Usage

1. **Upload Documents**: Drag PDF files to the upload area or click to browse
2. **View Library**: See all uploaded documents in the left panel
3. **Chat**: Ask questions about your documents in the chat area
4. **Voice Input**: Click the microphone button for speech-to-text
5. **System Status**: Monitor backend connectivity in real-time

## Browser Compatibility

- Modern browsers with ES6+ support
- Speech recognition requires Chrome/Edge for voice input
- No additional dependencies or build process required

## Notes

- Designed to seamlessly replace the existing React frontend
- Maintains all current Avi functionality
- Uses vanilla JavaScript for maximum compatibility
- Follows the exact API specifications from the integration guide