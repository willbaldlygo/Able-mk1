class AbleApp {
    constructor() {
        this.baseUrl = 'http://localhost:8000';
        this.documents = [];
        this.chatHistory = [];
        this.isRecording = false;

        this.initializeElements();
        this.setupEventListeners();
        this.checkSystemHealth();
        this.loadDocuments();
        this.loadAvailableModels();
        this.loadCurrentModel();
        this.checkMCPStatus();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.libraryList = document.getElementById('libraryList');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.voiceButton = document.getElementById('voiceButton');
        this.modelSelector = document.getElementById('modelSelector');
        this.modelDropdown = document.getElementById('modelDropdown');
        this.shutdownButton = document.getElementById('shutdownButton');


        // MCP elements
        this.mcpToggle = document.getElementById('mcpToggle');
        this.mcpModal = document.getElementById('mcpModal');
        this.mcpModalClose = document.getElementById('mcpModalClose');
        this.mcpModalCancel = document.getElementById('mcpModalCancel');
        this.mcpModalSave = document.getElementById('mcpModalSave');
        this.filesystemRoot = document.getElementById('filesystemRoot');
        this.gitRepos = document.getElementById('gitRepos');
        this.sqliteDbs = document.getElementById('sqliteDbs');

        // Visual modal elements
        this.visualModal = document.getElementById('visualModal');
        this.visualModalClose = document.getElementById('visualModalClose');
        this.visualModalClose2 = document.getElementById('visualModalClose2');
        this.visualModalContent = document.getElementById('visualModalContent');
    }

    setupEventListeners() {
        // Upload area
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.style.background = '#E8F5E8';
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.style.background = 'white';
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.style.background = 'white';
            this.handleFileUpload(e.dataTransfer.files);
        });

        // File input
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Chat
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        this.voiceButton.addEventListener('click', () => this.toggleVoiceRecording());
        
        // Model selector
        this.modelSelector.addEventListener('click', () => this.toggleModelDropdown());
        document.addEventListener('click', (e) => {
            if (!this.modelSelector.contains(e.target)) {
                this.modelDropdown.style.display = 'none';
            }
        });
        
        // Shutdown button
        this.shutdownButton.addEventListener('click', () => this.shutdownServices());

        // MCP events
        this.mcpToggle.addEventListener('click', () => this.toggleMCP());
        this.mcpModalClose.addEventListener('click', () => this.closeMCPModal());
        this.mcpModalCancel.addEventListener('click', () => this.closeMCPModal());
        this.mcpModalSave.addEventListener('click', () => this.saveMCPConfig());
        
        // Close modal on outside click
        this.mcpModal.addEventListener('click', (e) => {
            if (e.target === this.mcpModal) this.closeMCPModal();
        });

        // Visual modal events
        this.visualModalClose.addEventListener('click', () => this.closeVisualModal());
        this.visualModalClose2.addEventListener('click', () => this.closeVisualModal());
        this.visualModal.addEventListener('click', (e) => {
            if (e.target === this.visualModal) this.closeVisualModal();
        });
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            const data = await response.json();
            
            this.statusIndicator.classList.add('online');
            this.statusText.textContent = `Online - ${data.documents_count || 0} documents`;
        } catch (error) {
            this.statusIndicator.classList.remove('online');
            this.statusText.textContent = 'Offline - Backend not responding';
        }
    }

    async loadDocuments() {
        try {
            const response = await fetch(`${this.baseUrl}/documents`);
            this.documents = await response.json();
            this.renderLibrary();
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }

    renderLibrary() {
        this.libraryList.innerHTML = '';

        if (this.documents.length === 0) {
            this.libraryList.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No documents uploaded</div>';
            return;
        }

        this.documents.forEach(doc => {
            const item = document.createElement('div');
            item.className = 'library-item';
            const title = doc.name || doc.title || 'Untitled Document';
            const summary = doc.summary ? doc.summary.substring(0, 100) + '...' : 'No summary available';

            // Check for multimodal indicators
            const isMultimodal = doc.multimodal_info && (doc.multimodal_info.has_images || doc.multimodal_info.images_count > 0);
            const imageCount = doc.multimodal_info?.images_count || 0;
            const visualDescriptions = doc.multimodal_info?.visual_descriptions_count || 0;

            // Generate multimodal badges and visual context button
            let badges = '';
            let visualButton = '';
            if (isMultimodal) {
                badges += '<span class="multimodal-badge">🖼️ Multimodal</span>';
                if (imageCount > 0) {
                    badges += `<span class="image-count-badge">${imageCount} images</span>`;
                    visualButton = `<button class="visual-context-button" onclick="app.showVisualContext('${doc.id}')">View Images</button>`;
                }
                if (visualDescriptions > 0) {
                    badges += `<span class="visual-desc-badge">👁️ ${visualDescriptions} descriptions</span>`;
                }
            }

            item.innerHTML = `
                <div style="flex: 1;">
                    <div class="doc-title-row">
                        <span style="font-weight: 600; margin-bottom: 4px;">${title}</span>
                        ${badges}
                    </div>
                    <div style="font-size: 11px; color: #888; line-height: 1.3; margin-bottom: 4px;">${summary}</div>
                    <div style="font-size: 12px; color: #666; display: flex; align-items: center; gap: 8px;">
                        <span>${doc.chunk_count} chunks</span>
                        ${visualButton}
                    </div>
                </div>
                <button class="delete-btn" onclick="app.deleteDocument('${doc.id}')">×</button>
            `;
            this.libraryList.appendChild(item);
        });
    }

    async handleFileUpload(files) {
        const validFiles = Array.from(files).filter(file => {
            const isPdf = file.type === 'application/pdf';
            const isImage = ['image/png', 'image/jpg', 'image/jpeg'].includes(file.type);
            return isPdf || isImage;
        });

        if (validFiles.length === 0) {
            alert('Please select PDF files or images (PNG, JPG, JPEG) only');
            return;
        }

        // Check file types for appropriate processing message
        const hasImages = validFiles.some(file => file.type.startsWith('image/'));
        const hasPdfs = validFiles.some(file => file.type === 'application/pdf');

        this.uploadArea.classList.add('loading');

        // Show appropriate processing message
        if (hasImages && hasPdfs) {
            this.uploadArea.innerHTML = '<div class="spinner"></div> Processing PDFs + Images with visual analysis...';
        } else if (hasImages) {
            this.uploadArea.innerHTML = '<div class="spinner"></div> Processing Images with visual analysis...';
        } else {
            this.uploadArea.innerHTML = '<div class="spinner"></div> Processing PDFs with visual analysis...';
        }

        try {
            const formData = new FormData();
            validFiles.forEach(file => formData.append('files', file));

            // Always use multimodal endpoint for enhanced processing
            const endpoint = '/upload/multimodal';

            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                await this.loadDocuments();
                await this.checkSystemHealth();

                // Enhanced feedback for multimodal processing
                if (result.multimodal_info) {
                    const info = result.multimodal_info;
                    this.addChatMessage('system', `✅ ${result.message}\n🔍 Processed: ${info.images_processed || 0} images, ${info.visual_descriptions_generated || 0} visual descriptions`);
                } else {
                    this.addChatMessage('system', `✅ ${result.message}`);
                }
            } else {
                throw new Error(result.message || 'Upload failed');
            }
        } catch (error) {
            this.addChatMessage('system', `❌ Upload failed: ${error.message}`);
        } finally {
            this.uploadArea.classList.remove('loading');
            this.uploadArea.innerHTML = '<span class="icon">⬆</span><span>Drop PDFs and images here or click to browse</span>';
            this.fileInput.value = '';
        }
    }

    async deleteDocument(docId) {
        if (!confirm('Delete this document?')) return;

        try {
            const response = await fetch(`${this.baseUrl}/documents/${docId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                await this.loadDocuments();
                await this.checkSystemHealth();
                this.addChatMessage('system', '🗑️ Document deleted');
            } else {
                throw new Error('Delete failed');
            }
        } catch (error) {
            this.addChatMessage('system', `❌ Delete failed: ${error.message}`);
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        this.addChatMessage('user', message);
        this.chatInput.value = '';
        this.sendButton.disabled = true;

        // Add thinking indicator
        const thinkingMessage = this.addThinkingMessage();

        try {
            // Use staged reasoning for complex questions
            const useStaged = message.split(' ').length > 10 ||
                             /\b(how|why|what|when|where|explain|analyze|compare|discuss)\b/i.test(message);

            // Always use enhanced multimodal endpoints
            const endpoint = useStaged ? '/chat/enhanced' : '/chat/multimodal';

            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: message })
            });

            const result = await response.json();

            // Remove thinking indicator
            this.removeThinkingMessage(thinkingMessage);

            if (response.ok) {
                this.addChatMessage('assistant', result.answer, result.sources, result.multimodal_info);
            } else {
                throw new Error(result.detail || result.message || 'Chat failed');
            }
        } catch (error) {
            console.error('Chat error:', error);
            // Remove thinking indicator on error
            this.removeThinkingMessage(thinkingMessage);
            const errorMessage = error.message || 'Unknown error occurred';
            this.addChatMessage('system', `❌ Error: ${errorMessage}`);
        } finally {
            this.sendButton.disabled = false;
        }
    }

    addChatMessage(type, content, sources = [], multimodalInfo = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        // Render markdown content if it's from assistant, plain text for user
        let contentHtml;
        if (type === 'assistant' && typeof marked !== 'undefined') {
            // Configure marked for safe HTML rendering
            marked.setOptions({
                breaks: true,
                gfm: true
            });
            contentHtml = marked.parse(content);
        } else {
            contentHtml = content;
        }

        let html = `<div>${contentHtml}</div>`;

        // Add multimodal processing indicator if available
        if (multimodalInfo && type === 'assistant') {
            html += '<div class="multimodal-info">';
            if (multimodalInfo.visual_elements_used > 0) {
                html += `<span class="visual-indicator">👁️ Used ${multimodalInfo.visual_elements_used} visual elements</span>`;
            }
            if (multimodalInfo.processing_type) {
                html += `<span class="processing-type">🤖 ${multimodalInfo.processing_type}</span>`;
            }
            html += '</div>';
        }

        if (sources && sources.length > 0) {
            html += '<div class="message-sources">';
            sources.forEach(source => {
                let sourceIcon = '📄';
                let extraInfo = '';

                // Show visual indicators for sources with images
                if (source.has_visual_content) {
                    sourceIcon = '🖼️';
                    extraInfo = ' 👁️';
                }

                html += `<div class="source-item">${sourceIcon} ${source.document_name} (${Math.round(source.relevance_score * 100)}%)${extraInfo}</div>`;
            });
            html += '</div>';
        }

        messageDiv.innerHTML = html;
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    addThinkingMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant thinking';
        
        // Create animated thinking indicator
        const thinkingContent = document.createElement('div');
        thinkingContent.className = 'thinking-content';
        thinkingContent.innerHTML = `
            <div class="thinking-text">
                <span class="thinking-icon">🤔</span>
                <span>Able is thinking</span>
                <span class="thinking-dots">
                    <span>.</span>
                    <span>.</span>
                    <span>.</span>
                </span>
            </div>
        `;
        
        messageDiv.appendChild(thinkingContent);
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        return messageDiv;
    }

    removeThinkingMessage(thinkingElement) {
        if (thinkingElement && thinkingElement.parentNode) {
            thinkingElement.parentNode.removeChild(thinkingElement);
        }
    }

    toggleVoiceRecording() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert('Speech recognition not supported in this browser');
            return;
        }

        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    startRecording() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isRecording = true;
            this.voiceButton.classList.add('recording');
            this.voiceButton.textContent = '🔴';
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.chatInput.value = transcript;
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.stopRecording();
        };

        this.recognition.onend = () => {
            this.stopRecording();
        };

        this.recognition.start();
    }

    stopRecording() {
        if (this.recognition) {
            this.recognition.stop();
        }
        
        this.isRecording = false;
        this.voiceButton.classList.remove('recording');
        this.voiceButton.textContent = '🎤';
    }

    async loadAvailableModels() {
        try {
            const response = await fetch(`${this.baseUrl}/models/available`);
            const data = await response.json();
            this.availableModels = data.models || [];
            this.renderModelDropdown();
        } catch (error) {
            console.error('Failed to load available models:', error);
        }
    }

    async loadCurrentModel() {
        try {
            // Check localStorage for saved model preference first
            const savedModel = localStorage.getItem('able_selected_model');
            if (savedModel) {
                const modelData = JSON.parse(savedModel);
                // Try to switch to saved model
                const switchResponse = await fetch(`${this.baseUrl}/models/switch`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        provider: modelData.provider,
                        model: modelData.name 
                    })
                });
                
                if (switchResponse.ok) {
                    this.currentModel = modelData;
                    this.updateModelSelectorText();
                    return;
                }
            }
            
            // Fallback to backend current model
            const response = await fetch(`${this.baseUrl}/models/current`);
            const data = await response.json();
            this.currentModel = data.model;
            this.updateModelSelectorText();
        } catch (error) {
            console.error('Failed to load current model:', error);
            this.currentModel = { display_name: 'Claude 3.5 Sonnet', name: 'claude-3-5-sonnet-20241022' };
            this.updateModelSelectorText();
        }
    }

    updateModelSelectorText() {
        if (this.currentModel) {
            this.modelSelector.textContent = this.currentModel.display_name || this.currentModel.name;
        }
    }

    renderModelDropdown() {
        if (!this.availableModels || this.availableModels.length === 0) return;
        
        this.modelDropdown.innerHTML = '';
        
        this.availableModels.forEach(model => {
            const item = document.createElement('div');
            item.className = 'model-dropdown-item';
            item.textContent = model.display_name || model.name;
            item.addEventListener('click', () => this.selectModel(model));
            this.modelDropdown.appendChild(item);
        });
    }

    toggleModelDropdown() {
        const isVisible = this.modelDropdown.style.display === 'block';
        this.modelDropdown.style.display = isVisible ? 'none' : 'block';
    }

    async selectModel(model) {
        try {
            const response = await fetch(`${this.baseUrl}/models/switch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    provider: model.provider,
                    model: model.name 
                })
            });

            if (response.ok) {
                this.currentModel = model;
                this.updateModelSelectorText();
                this.modelDropdown.style.display = 'none';
                
                // Save model selection to localStorage for persistence
                localStorage.setItem('able_selected_model', JSON.stringify({
                    name: model.name,
                    display_name: model.display_name,
                    provider: model.provider
                }));
                
                this.addChatMessage('system', `🔄 Switched to ${model.display_name || model.name}`);
            } else {
                throw new Error('Failed to switch model');
            }
        } catch (error) {
            this.addChatMessage('system', `❌ Failed to switch model: ${error.message}`);
        }
    }
    
    async shutdownServices() {
        if (!confirm('Are you sure you want to shutdown all Able services? This will close the application.')) {
            return;
        }
        
        try {
            this.addChatMessage('system', '🔴 Shutting down Able services...');
            
            const response = await fetch(`${this.baseUrl}/shutdown`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.addChatMessage('system', `✅ ${data.message}`);
                this.addChatMessage('system', `📊 Session completed: ${data.session_log.documents_processed} documents processed`);
                
                // Disable UI after shutdown
                setTimeout(() => {
                    document.body.innerHTML = `
                        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;">
                            <h1 style="color: #666;">🔴 Able Services Shutdown</h1>
                            <p style="color: #999;">All services have been cleanly terminated.</p>
                            <p style="color: #999; font-size: 14px;">You can close this browser window.</p>
                        </div>
                    `;
                }, 2000);
            } else {
                throw new Error('Shutdown request failed');
            }
        } catch (error) {
            this.addChatMessage('system', `❌ Shutdown failed: ${error.message}`);
        }
    }
    
    // MCP Methods
    async checkMCPStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/mcp/status`);
            const status = await response.json();
            
            if (status.enabled && status.session_active) {
                this.mcpToggle.classList.add('active');
                this.mcpToggle.textContent = '🔧 MCP ON';
            } else {
                this.mcpToggle.classList.remove('active');
                this.mcpToggle.textContent = '🔧 MCP';
            }
        } catch (error) {
            console.error('Failed to check MCP status:', error);
        }
    }
    
    async toggleMCP() {
        try {
            const response = await fetch(`${this.baseUrl}/mcp/status`);
            const status = await response.json();
            
            if (status.enabled && status.session_active) {
                // Disable MCP
                await this.disableMCP();
            } else {
                // Enable MCP - show configuration modal
                this.showMCPModal();
            }
        } catch (error) {
            console.error('Failed to toggle MCP:', error);
            this.addChatMessage('system', '❌ Failed to toggle MCP');
        }
    }
    
    async disableMCP() {
        try {
            const response = await fetch(`${this.baseUrl}/mcp/toggle`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: false })
            });
            
            if (response.ok) {
                await this.checkMCPStatus();
                this.addChatMessage('system', '🔧 MCP disabled');
            } else {
                throw new Error('Failed to disable MCP');
            }
        } catch (error) {
            this.addChatMessage('system', `❌ Failed to disable MCP: ${error.message}`);
        }
    }
    
    showMCPModal() {
        // Set default values
        this.filesystemRoot.value = '/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data';
        this.gitRepos.value = '/Users/will/AVI BUILD/Able3_Main_WithVoiceMode';
        this.sqliteDbs.value = '/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data/vectordb';
        
        this.mcpModal.style.display = 'block';
    }
    
    closeMCPModal() {
        this.mcpModal.style.display = 'none';
    }
    
    async saveMCPConfig() {
        try {
            // First enable MCP
            const toggleResponse = await fetch(`${this.baseUrl}/mcp/toggle`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: true })
            });
            
            if (!toggleResponse.ok) {
                throw new Error('Failed to enable MCP session');
            }
            
            // Then configure it
            const config = {
                filesystem_root: this.filesystemRoot.value.trim() || null,
                git_repositories: this.gitRepos.value.trim() ? 
                    this.gitRepos.value.split('\n').map(line => line.trim()).filter(line => line) : [],
                sqlite_connections: this.sqliteDbs.value.trim() ? 
                    this.sqliteDbs.value.split('\n').map(line => line.trim()).filter(line => line) : [],
                enabled_tools: []
            };
            
            const configResponse = await fetch(`${this.baseUrl}/mcp/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            const result = await configResponse.json();
            
            if (configResponse.ok) {
                this.closeMCPModal();
                await this.checkMCPStatus();
                this.addChatMessage('system', `🔧 MCP enabled with ${result.started_servers.length} servers: ${result.started_servers.join(', ')}`);
            } else {
                throw new Error(result.detail || 'Failed to configure MCP');
            }
        } catch (error) {
            this.addChatMessage('system', `❌ Failed to configure MCP: ${error.message}`);
        }
    }


    // Visual Context Methods
    async showVisualContext(documentId) {
        try {
            const response = await fetch(`${this.baseUrl}/documents/${documentId}/processing-info`);
            const info = await response.json();

            if (!info.multimodal_info || !info.multimodal_info.images) {
                this.addChatMessage('system', '❌ No visual content found for this document');
                return;
            }

            const images = info.multimodal_info.images;
            let thumbnailsHtml = '<div class="visual-thumbnails">';

            images.forEach((image, index) => {
                const imageUrl = `${this.baseUrl}/sources/images/${image.filename}`;
                thumbnailsHtml += `
                    <div class="visual-thumbnail">
                        <div class="filename">${image.filename}</div>
                        <img src="${imageUrl}" alt="Document image ${index + 1}" onclick="window.open('${imageUrl}', '_blank')">
                        <div class="description">${image.description || 'No description available'}</div>
                    </div>
                `;
            });

            thumbnailsHtml += '</div>';
            this.visualModalContent.innerHTML = thumbnailsHtml;
            this.visualModal.style.display = 'block';

        } catch (error) {
            console.error('Failed to load visual context:', error);
            this.addChatMessage('system', `❌ Failed to load visual context: ${error.message}`);
        }
    }

    closeVisualModal() {
        this.visualModal.style.display = 'none';
        this.visualModalContent.innerHTML = '';
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AbleApp();
});

// Health check every 30 seconds
setInterval(() => {
    if (window.app) {
        window.app.checkSystemHealth();
    }
}, 30000);