/**
 * Document management hook for Able3
 */
import { useState, useEffect } from 'react';
import { apiService } from '../services/api';

export const useDocuments = () => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadDocuments = async (retryCount = 0) => {
    try {
      setLoading(true);
      setError(null);
      const docs = await apiService.getDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Load documents error:', err);
      
      // Retry up to 3 times with increasing delay for initial load issues
      if (retryCount < 3 && (err.code === 'ECONNREFUSED' || err.response?.status >= 500)) {
        console.log(`Retrying document load (attempt ${retryCount + 1}/3)...`);
        setTimeout(() => {
          loadDocuments(retryCount + 1);
        }, 1000 * (retryCount + 1)); // 1s, 2s, 3s delays
        return;
      }
      
      setError('Failed to load documents');
    } finally {
      if (retryCount === 0) { // Only set loading false on the original call
        setLoading(false);
      }
    }
  };

  const uploadDocument = async (file) => {
    try {
      const result = await apiService.uploadDocument(file);
      if (result.success) {
        setDocuments(prev => [result.document, ...prev]);
        return { success: true, message: result.message };
      } else {
        return { success: false, message: result.message };
      }
    } catch (err) {
      console.error('Upload error:', err);
      return { 
        success: false, 
        message: err.response?.data?.detail || 'Upload failed' 
      };
    }
  };

  const deleteDocument = async (documentId) => {
    try {
      await apiService.deleteDocument(documentId);
      setDocuments(prev => prev.filter(doc => doc.id !== documentId));
      return { success: true };
    } catch (err) {
      console.error('Delete error:', err);
      return { 
        success: false, 
        message: err.response?.data?.detail || 'Delete failed' 
      };
    }
  };

  useEffect(() => {
    // Small delay to ensure backend is ready
    const timer = setTimeout(() => {
      loadDocuments();
    }, 500);
    
    return () => clearTimeout(timer);
  }, []);

  return {
    documents,
    loading,
    error,
    uploadDocument,
    deleteDocument,
    refreshDocuments: loadDocuments,
  };
};