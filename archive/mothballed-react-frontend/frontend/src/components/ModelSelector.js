/**
 * Model Selection Component for Able
 */
import React, { useState, useEffect } from 'react';
import { Settings, Cpu, Cloud, Zap, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { apiService } from '../services/api';

const ModelSelector = ({ vintage = false, onModelChange, currentModel }) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [pendingModel, setPendingModel] = useState(null);
  const [modelMetrics, setModelMetrics] = useState({});
  const [isExpanded, setIsExpanded] = useState(false);

  // Load models and metrics on mount
  useEffect(() => {
    loadModels();
    loadModelMetrics();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const response = await apiService.getAvailableModels();
      setModels(response.models || []);
    } catch (error) {
      console.error('Failed to load models:', error);
      // Fallback to mock data for development
      setModels([
        {
          id: 'claude-3-sonnet',
          name: 'Claude 3 Sonnet',
          type: 'cloud',
          status: 'online',
          description: 'Balanced performance and speed',
          capabilities: ['text', 'analysis', 'research'],
          responseTime: 1200
        },
        {
          id: 'claude-3-haiku',
          name: 'Claude 3 Haiku',
          type: 'cloud',
          status: 'online',
          description: 'Fast and efficient',
          capabilities: ['text', 'quick-response'],
          responseTime: 800
        },
        {
          id: 'local-llama',
          name: 'Llama 2 7B',
          type: 'local',
          status: 'offline',
          description: 'Local processing, privacy focused',
          capabilities: ['text', 'offline'],
          responseTime: 2400
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const loadModelMetrics = () => {
    // Load from localStorage or initialize
    const saved = localStorage.getItem('model-metrics');
    if (saved) {
      setModelMetrics(JSON.parse(saved));
    }
  };

  const saveModelMetrics = (metrics) => {
    localStorage.setItem('model-metrics', JSON.stringify(metrics));
    setModelMetrics(metrics);
  };

  const handleModelSelect = (model) => {
    if (model.id === currentModel?.id) return;
    
    if (model.status !== 'online') {
      alert(`${model.name} is currently ${model.status}. Please select an available model.`);
      return;
    }

    setPendingModel(model);
    setShowConfirmDialog(true);
  };

  const confirmModelSwitch = async () => {
    if (!pendingModel) return;

    try {
      setLoading(true);
      await apiService.setActiveModel(pendingModel.id);
      onModelChange?.(pendingModel);
      
      // Update metrics
      const now = Date.now();
      const newMetrics = {
        ...modelMetrics,
        [pendingModel.id]: {
          ...modelMetrics[pendingModel.id],
          lastUsed: now,
          switchCount: (modelMetrics[pendingModel.id]?.switchCount || 0) + 1
        }
      };
      saveModelMetrics(newMetrics);
      
      setShowConfirmDialog(false);
      setPendingModel(null);
    } catch (error) {
      console.error('Failed to switch model:', error);
      alert('Failed to switch model. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'offline':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'loading':
        return <div className="w-4 h-4 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const getTypeIcon = (type) => {
    return type === 'local' ? 
      <Cpu className="w-4 h-4" /> : 
      <Cloud className="w-4 h-4" />;
  };

  const formatResponseTime = (ms) => {
    return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
  };

  if (vintage) {
    return (
      <div className="vintage-model-selector model-selector">
        <div 
          className="vintage-title cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <Settings className="vintage-icon" style={{color: '#d4834a'}} />
          <div className="vintage-text">Model Selection</div>
          <div className="ml-auto vintage-status-indicator">
            {currentModel ? (
              <div className="flex items-center space-x-2">
                {getTypeIcon(currentModel.type)}
                <span className="text-xs">{currentModel.name}</span>
                {getStatusIcon(currentModel.status)}
              </div>
            ) : (
              <span className="text-xs opacity-60">No model selected</span>
            )}
          </div>
        </div>

        {isExpanded && (
          <div className="vintage-inner">
            <div className="space-y-2">
              {loading ? (
                <div className="text-center py-4">
                  <div className="vintage-loading-spinner mx-auto mb-2"></div>
                  <div className="text-xs vintage-text opacity-60">Loading models...</div>
                </div>
              ) : (
                models.map((model) => (
                  <div
                    key={model.id}
                    className={`vintage-model-card ${
                      currentModel?.id === model.id ? 'vintage-model-active' : ''
                    } ${model.status !== 'online' ? 'vintage-model-disabled' : 'cursor-pointer'}`}
                    onClick={() => handleModelSelect(model)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          {getTypeIcon(model.type)}
                          <div className="font-medium text-sm">{model.name}</div>
                          {getStatusIcon(model.status)}
                        </div>
                        <div className="text-xs opacity-75 mb-2">{model.description}</div>
                        <div className="flex items-center space-x-3 text-xs">
                          <div className="flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>{formatResponseTime(model.responseTime)}</span>
                          </div>
                          {modelMetrics[model.id]?.switchCount && (
                            <span className="opacity-60">
                              Used {modelMetrics[model.id].switchCount}x
                            </span>
                          )}
                        </div>
                      </div>
                      {currentModel?.id === model.id && (
                        <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Confirmation Dialog */}
        {showConfirmDialog && (
          <div className="vintage-overlay">
            <div className="vintage-dialog">
              <div className="vintage-dialog-header">
                <Settings className="vintage-icon" />
                <span>Switch Model</span>
              </div>
              <div className="vintage-dialog-content">
                <p>Switch to {pendingModel?.name}?</p>
                <p className="text-xs opacity-75 mt-2">
                  Current conversations will continue with the new model.
                </p>
              </div>
              <div className="vintage-dialog-actions">
                <button
                  className="vintage-btn-secondary"
                  onClick={() => setShowConfirmDialog(false)}
                  disabled={loading}
                >
                  Cancel
                </button>
                <button
                  className="vintage-btn-primary"
                  onClick={confirmModelSwitch}
                  disabled={loading}
                >
                  {loading ? 'Switching...' : 'Switch'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Regular glass card style for non-vintage mode
  return (
    <div className="glass-card">
      <div className="p-6 border-b-2 border-accent-sage">
        <div className="flex items-center space-x-4">
          <Settings className="h-7 w-7 text-primary-500" />
          <h2 className="text-2xl font-bold text-text-primary">Model Selection</h2>
        </div>
        <p className="text-base text-text-secondary mt-2 font-medium">
          Choose your AI model for research and analysis
        </p>
      </div>

      <div className="p-6">
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto mb-4"></div>
            <p className="text-text-secondary">Loading available models...</p>
          </div>
        ) : (
          <div className="space-y-4">
            {models.map((model) => (
              <div
                key={model.id}
                className={`border-2 rounded-card p-4 transition-all cursor-pointer ${
                  currentModel?.id === model.id
                    ? 'border-primary-500 bg-accent-beige'
                    : model.status === 'online'
                    ? 'border-accent-sage hover:border-primary-500 bg-glass-white'
                    : 'border-gray-300 bg-gray-50 cursor-not-allowed opacity-60'
                }`}
                onClick={() => handleModelSelect(model)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      {getTypeIcon(model.type)}
                      <h3 className="font-bold text-lg text-text-primary">{model.name}</h3>
                      {getStatusIcon(model.status)}
                      <span className={`text-sm px-2 py-1 rounded ${
                        model.type === 'local' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                      }`}>
                        {model.type}
                      </span>
                    </div>
                    <p className="text-text-secondary mb-3">{model.description}</p>
                    <div className="flex items-center space-x-4 text-sm">
                      <div className="flex items-center space-x-1 text-text-secondary">
                        <Clock className="w-4 h-4" />
                        <span>~{formatResponseTime(model.responseTime)}</span>
                      </div>
                      {model.capabilities.map((cap) => (
                        <span key={cap} className="px-2 py-1 bg-accent-beige text-text-primary rounded text-xs">
                          {cap}
                        </span>
                      ))}
                    </div>
                  </div>
                  {currentModel?.id === model.id && (
                    <CheckCircle className="w-6 h-6 text-primary-500 flex-shrink-0" />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Confirmation Dialog */}
      {showConfirmDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-glass-white border-2 border-accent-sage rounded-card p-6 max-w-md mx-4">
            <div className="flex items-center space-x-3 mb-4">
              <Settings className="h-6 w-6 text-primary-500" />
              <h3 className="text-xl font-bold text-text-primary">Switch Model</h3>
            </div>
            <p className="text-text-secondary mb-4">
              Switch to <strong>{pendingModel?.name}</strong>?
            </p>
            <p className="text-sm text-text-secondary mb-6">
              Current conversations will continue with the new model.
            </p>
            <div className="flex space-x-3 justify-end">
              <button
                className="px-4 py-2 border-2 border-accent-sage rounded-card hover:bg-accent-beige transition-colors"
                onClick={() => setShowConfirmDialog(false)}
                disabled={loading}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 bg-primary-500 text-text-light rounded-card hover:bg-accent-coral transition-colors disabled:opacity-50"
                onClick={confirmModelSwitch}
                disabled={loading}
              >
                {loading ? 'Switching...' : 'Switch Model'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;