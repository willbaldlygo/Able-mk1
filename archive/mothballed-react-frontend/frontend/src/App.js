/**
 * Able Main Application Component - Vintage IBM Style
 */
import React, { useState, useEffect } from 'react';
import { Globe, FileText, ChevronDown, Settings } from 'lucide-react';
import DocumentManager from './components/DocumentManager';
import ChatInterface from './components/ChatInterface';
import UrlInput from './components/UrlInput';
import { useDocuments } from './hooks/useDocuments';
import { apiService } from './services/api';
import './styles/globals.css';

function App() {
  const { documents, refreshDocuments } = useDocuments();
  const [urlLoading, setUrlLoading] = useState(false);
  const [currentModel, setCurrentModel] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);

  // Load current model and available models on mount
  useEffect(() => {
    loadCurrentModel();
    loadAvailableModels();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showModelDropdown && !event.target.closest('.relative')) {
        setShowModelDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showModelDropdown]);

  const loadCurrentModel = async () => {
    try {
      const response = await apiService.getCurrentModel();
      setCurrentModel(response.model);
    } catch (error) {
      console.error('Failed to load current model:', error);
      // Set a default fallback model
      setCurrentModel({
        id: 'claude-3-sonnet',
        name: 'Claude 3 Sonnet',
        type: 'cloud',
        status: 'online',
        responseTime: 1200
      });
    }
  };

  const loadAvailableModels = async () => {
    try {
      setLoadingModels(true);
      const response = await apiService.getAvailableModels();
      // The API returns { models: [...], providers: {...} }
      // We want the flat models array
      const models = response.models || [];
      setAvailableModels(models);
    } catch (error) {
      console.error('Failed to load available models:', error);
      setAvailableModels([]);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleModelChange = async (model) => {
    try {
      setShowModelDropdown(false);
      await apiService.switchModel(model.provider, model.name);
      setCurrentModel({
        ...model,
        id: model.name // Use name as id for compatibility
      });
      setModelError(null);
    } catch (error) {
      console.error('Failed to switch model:', error);
      setModelError({
        model: model,
        error: error.message
      });
    }
  };

  const handleUrlSubmit = async (urlData) => {
    setUrlLoading(true);
    try {
      const result = await apiService.scrapeUrl(urlData);
      if (result.success) {
        await refreshDocuments();
      }
      return result;
    } catch (error) {
      throw error;
    } finally {
      setUrlLoading(false);
    }
  };


  const handleModelError = (model, error) => {
    setModelError({ model, error, timestamp: Date.now() });
    // You could implement automatic fallback here
    console.error(`Model ${model.name} encountered an error:`, error);
  };

  return (
    <div className="min-h-screen vintage-background">
      {/* Header */}
      <header className="vintage-header">
        <div className="vintage-header-content">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-card shadow-card overflow-hidden">
                <img src="/favicon.png" alt="Able" className="w-full h-full object-cover" />
              </div>
              <h1 className="text-2xl font-bold vintage-text gradient-text">Able mk I</h1>
            </div>
            <div className="text-base vintage-text font-medium">
              AI Research Assistant
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3 text-base vintage-text font-medium">
              <FileText className="w-5 h-5" style={{color: '#8b3a3a'}} />
              <span>{documents.length} document{documents.length !== 1 ? 's' : ''}</span>
            </div>
            {/* Model Selector Dropdown */}
            <div className="relative">
              <button
                className="flex items-center space-x-2 text-sm vintage-text hover:bg-orange-50 hover:bg-opacity-20 px-3 py-1 rounded transition-colors"
                onClick={() => setShowModelDropdown(!showModelDropdown)}
              >
                <Settings className="w-4 h-4" style={{color: '#d4834a'}} />
                {currentModel ? (
                  <>
                    <span>{currentModel.type === 'local' ? '💻' : '☁️'}</span>
                    <span>{currentModel.name}</span>
                    <div className={`w-2 h-2 rounded-full ${
                      currentModel.status === 'online' ? 'bg-green-500' :
                      currentModel.status === 'loading' ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}></div>
                  </>
                ) : (
                  <span>Select Model</span>
                )}
                <ChevronDown className={`w-4 h-4 transition-transform ${showModelDropdown ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              {showModelDropdown && (
                <div className="absolute top-full right-0 mt-2 w-80 bg-amber-50 border-2 border-orange-200 rounded-lg shadow-lg z-50">
                  <div className="p-3 border-b border-orange-200">
                    <div className="flex items-center space-x-2 text-sm font-medium vintage-text">
                      <Settings className="w-4 h-4" style={{color: '#d4834a'}} />
                      <span>Select AI Model</span>
                    </div>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    {loadingModels ? (
                      <div className="p-4 text-center text-sm vintage-text opacity-60">
                        Loading models...
                      </div>
                    ) : availableModels.length === 0 ? (
                      <div className="p-4 text-center text-sm vintage-text opacity-60">
                        No models available
                      </div>
                    ) : (
                      availableModels.map((model) => (
                        <button
                          key={`${model.provider}-${model.name}`}
                          className={`w-full text-left p-3 hover:bg-orange-100 border-b border-orange-100 transition-colors ${
                            currentModel?.name === model.name ? 'bg-orange-50' : ''
                          }`}
                          onClick={() => handleModelChange(model)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-1">
                                <span>{model.type === 'local' ? '💻' : '☁️'}</span>
                                <span className="font-medium text-sm vintage-text">{model.display_name || model.name}</span>
                                {model.available ? (
                                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                                ) : (
                                  <div className="w-2 h-2 rounded-full bg-red-500"></div>
                                )}
                              </div>
                              <div className="text-xs opacity-75 vintage-text">
                                {model.provider} • {model.type}
                              </div>
                            </div>
                            {currentModel?.name === model.name && (
                              <div className="text-green-600">✓</div>
                            )}
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
            <div className="w-4 h-4 bg-gradient-to-r from-green-400 to-green-600 rounded-full shadow-sm" title="System online"></div>
          </div>
        </div>
      </header>

      {/* Vintage 2x2 Grid Layout */}
      <div className="vintage-layout">
        {/* Top Left - Add Web Content */}
        <div className="vintage-panel web-content-card">
          <div className="vintage-title">
            <Globe className="vintage-icon" style={{color: '#1e3a5f'}} />
            <div className="vintage-text">Add Web Content</div>
          </div>
          <div className="vintage-inner">
            <UrlInput onUrlSubmit={handleUrlSubmit} loading={urlLoading} vintage={true} />
          </div>
        </div>

        {/* Top Right - System Status */}
        <div className="vintage-panel questions-card">
          <div className="vintage-title">
            <div className="w-4 h-4 bg-gradient-to-r from-green-400 to-green-600 rounded-full shadow-sm"></div>
            <div className="vintage-text">System Status</div>
          </div>
          <div className="vintage-inner">
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="vintage-text opacity-75">Current Model:</span>
                <div className="flex items-center space-x-2">
                  {currentModel ? (
                    <>
                      <span>{currentModel.type === 'local' ? '💻' : '☁️'}</span>
                      <span className="vintage-text font-medium">{currentModel.name}</span>
                      <div className={`w-2 h-2 rounded-full ${
                        currentModel.status === 'online' ? 'bg-green-500' :
                        currentModel.status === 'loading' ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}></div>
                    </>
                  ) : (
                    <span className="vintage-text opacity-60">None selected</span>
                  )}
                </div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className="vintage-text opacity-75">AI Providers:</span>
                <div className="flex items-center space-x-2">
                  <span className="text-xs vintage-text">☁️ Anthropic</span>
                  <span className="text-xs vintage-text">💻 Ollama</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className="vintage-text opacity-75">Documents:</span>
                <span className="vintage-text font-medium">{documents.length} loaded</span>
              </div>
              
              {modelError && (
                <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-xs">
                  <span className="text-red-600">⚠️ {modelError.model?.name} error</span>
                  <div className="mt-1 opacity-75">{modelError.error}</div>
                </div>
              )}
              
              <div className="mt-4 pt-3 border-t border-orange-200">
                <div className="text-xs vintage-text opacity-60 text-center">
                  Switch models using the header dropdown ↑
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Left - Document Library */}
        <div className="vintage-panel library-card">
          <div className="vintage-title">
            <FileText className="vintage-icon" style={{color: '#8b3a3a'}} />
            <div className="vintage-text">Document Library</div>
          </div>
          <div className="vintage-inner">
            <DocumentManager vintage={true} />
          </div>
        </div>

        {/* Bottom Right - Chat Interface */}
        <div className="vintage-panel input-card">
          <div className="vintage-inner chat-container">
            <ChatInterface 
              hasDocuments={documents.length > 0} 
              vintage={true}
              currentModel={currentModel}
              onModelError={handleModelError}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;