/**
 * Clean API service for Able3
 */
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '',
  timeout: 90000, // 90 seconds for large local models like gpt-oss:20b
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Document operations
  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },

  async getDocuments() {
    const response = await api.get('/documents');
    return response.data;
  },

  async deleteDocument(documentId) {
    const response = await api.delete(`/documents/${documentId}`);
    return response.data;
  },

  async scrapeUrl(urlData) {
    const response = await api.post('/scrape-url', urlData);
    return response.data;
  },

  // Chat operations
  async chatWithDocuments(question, documentIds = null) {
    const response = await api.post('/chat', {
      question,
      document_ids: documentIds,
    });
    return response.data;
  },

  // System operations
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Model management operations
  async getAvailableModels() {
    const response = await api.get('/models/available');
    return response.data;
  },

  async getCurrentModel() {
    const response = await api.get('/models/current');
    return response.data;
  },

  async setActiveModel(modelId) {
    const response = await api.post('/models/set-active', {
      model_id: modelId
    });
    return response.data;
  },

  async switchModel(provider, modelName) {
    const response = await api.post('/models/switch', {
      provider: provider,
      model: modelName
    });
    return response.data;
  },

  async getModelStatus(modelId) {
    const response = await api.get(`/models/${modelId}/status`);
    return response.data;
  },

  async getModelMetrics() {
    const response = await api.get('/models/metrics');
    return response.data;
  },
};

export default apiService;