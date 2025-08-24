/**
 * Enhanced Document Manager for Able3
 */
import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, Trash2, Clock, FileText, Globe, ExternalLink } from 'lucide-react';
import { useDocuments } from '../hooks/useDocuments';
import UrlInput from './UrlInput';
import { apiService } from '../services/api';

const DocumentManager = ({ vintage = false }) => {
  const { documents, loading, uploadDocument, deleteDocument, refreshDocuments } = useDocuments();
  const [uploading, setUploading] = useState(false);
  const [scrapingUrl, setScrapingUrl] = useState(false);
  const [message, setMessage] = useState(null);

  const onDrop = async (acceptedFiles) => {
    setUploading(true);
    
    // Support batch upload
    const uploadPromises = acceptedFiles.map(async (file) => {
      const result = await uploadDocument(file);
      return { file: file.name, ...result };
    });

    try {
      const results = await Promise.all(uploadPromises);
      
      const successful = results.filter(r => r.success);
      const failed = results.filter(r => !r.success);

      if (successful.length > 0) {
        setMessage({
          type: 'success',
          text: `Successfully uploaded ${successful.length} document(s)`
        });
      }

      if (failed.length > 0) {
        // Show specific error messages for failed uploads
        const errorDetails = failed.map(f => `${f.file}: ${f.message || 'Unknown error'}`);
        setMessage({
          type: 'error',
          text: failed.length === 1 
            ? errorDetails[0]
            : `Upload failures:\n${errorDetails.join('\n')}`
        });
      }

    } catch (error) {
      setMessage({
        type: 'error',
        text: 'Upload failed'
      });
    } finally {
      setUploading(false);
    }
  };

  const handleUrlSubmit = async (urlData) => {
    setScrapingUrl(true);
    
    try {
      const result = await apiService.scrapeUrl(urlData);
      
      if (result.success) {
        setMessage({
          type: 'success',
          text: result.message
        });
        
        // Refresh documents list
        await refreshDocuments();
      } else {
        setMessage({
          type: 'error',
          text: result.message || 'Failed to process URL'
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || error.message || 'Failed to process URL'
      });
    } finally {
      setScrapingUrl(false);
    }
  };

  const onDropRejected = (fileRejections) => {
    const rejectedFiles = fileRejections.map(fr => fr.file.name);
    setMessage({
      type: 'error',
      text: `Only PDF files are allowed. Rejected files: ${rejectedFiles.join(', ')}`
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true,
    maxSize: 50 * 1024 * 1024, // 50MB limit
  });

  const handleDelete = async (docId, docName) => {
    if (window.confirm(`Delete "${docName}"?`)) {
      const result = await deleteDocument(docId);
      if (result.success) {
        setMessage({
          type: 'success',
          text: 'Document deleted successfully'
        });
      } else {
        setMessage({
          type: 'error',
          text: result.message || 'Delete failed'
        });
      }
    }
  };

  const formatFileSize = (bytes) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (vintage) {
    return (
      <div className="flex flex-col h-full space-y-4">
        {/* Upload Area */}
        <div
          {...getRootProps()}
          className={`vintage-upload cursor-pointer transition-all duration-300 ${
            isDragActive ? 'vintage-upload-active' : ''
          }`}
        >
          <input {...getInputProps()} />
          <div className="upload-content">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{color: '#7d5a3a'}}>
              <path d="M4 17h16"/>
              <path d="M12 3v14"/>
              <path d="M8 11l4-4 4 4"/>
            </svg>
            <div>
              {uploading 
                ? 'Processing documents...'
                : isDragActive
                ? 'Drop PDF files here...'
                : 'Drag & drop PDF files here, or click to select'
              }
            </div>
          </div>
          <div className="upload-subtitle">
            Supports multiple files • Max 50MB per file
          </div>
        </div>

        {/* Document List - Full Information */}
        {documents.length > 0 && (
          <div className="flex-1 overflow-y-auto space-y-3 max-h-96">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="vintage-document-item p-3"
              >
                <div className="flex items-start justify-between space-x-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-2">
                      {doc.source_type === 'web' ? (
                        <Globe className="h-4 w-4 flex-shrink-0" style={{color: '#1e3a5f'}} />
                      ) : (
                        <File className="h-4 w-4 flex-shrink-0" style={{color: '#8b3a3a'}} />
                      )}
                      <h4 
                        className="vintage-text text-sm font-bold flex-1 leading-tight"
                        title={doc.name}
                        style={{
                          wordWrap: 'break-word',
                          overflowWrap: 'break-word',
                          hyphens: 'auto'
                        }}
                      >
                        {doc.name}
                      </h4>
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        doc.source_type === 'web' 
                          ? 'bg-blue-100 text-blue-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {doc.source_type === 'web' ? 'WEB' : 'PDF'}
                      </span>
                    </div>
                    
                    <p 
                      className="vintage-text text-xs mb-2 line-clamp-2"
                      title={doc.summary}
                      style={{
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        wordWrap: 'break-word'
                      }}
                    >
                      {doc.summary}
                    </p>
                    
                    <div className="flex items-center space-x-3 text-xs vintage-text opacity-75">
                      <span className="flex items-center">
                        <Clock className="h-3 w-3 mr-1" />
                        {formatDate(doc.created_at)}
                      </span>
                      <span>{formatFileSize(doc.file_size)}</span>
                      <span>{doc.chunk_count} excerpts</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => handleDelete(doc.id, doc.name)}
                    className="vintage-delete-btn flex-shrink-0"
                    title="Delete document"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Message Display */}
        {message && (
          <div className={`vintage-message ${message.type === 'success' ? 'vintage-message-success' : 'vintage-message-error'}`}>
            <span className="text-xs vintage-text">{message.text}</span>
            <button
              onClick={() => setMessage(null)}
              className="text-xs underline hover:no-underline ml-2"
            >
              ×
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* URL Input */}
      <UrlInput 
        onUrlSubmit={handleUrlSubmit} 
        loading={scrapingUrl || uploading || loading}
      />

      {/* Upload Area */}
      <div className="glass-card">
        <div className="p-6">
          <h2 className="text-2xl font-bold text-primary-500 mb-6">
            Document Library
          </h2>
          
          <div
            {...getRootProps()}
            className={`border-4 border-dashed rounded-card p-12 text-center cursor-pointer transition-all duration-300 ${
              isDragActive
                ? 'border-primary-500 bg-glass-light shadow-glass'
                : 'border-accent-sage hover:border-primary-500 bg-glass-white hover:shadow-glass upload-area'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-16 w-16 text-primary-500 mb-6" />
            {uploading ? (
              <div className="text-primary-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-4 border-primary-500 mx-auto mb-4"></div>
                <p className="text-lg font-bold">Processing documents...</p>
              </div>
            ) : (
              <div>
                <p className="text-xl text-text-primary mb-3 font-bold">
                  {isDragActive
                    ? 'Drop PDF files here...'
                    : 'Drag & drop PDF files here, or click to select'}
                </p>
                <p className="text-base text-text-secondary font-medium">
                  Supports multiple files • Max 50MB per file
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Message Display */}
      {message && (
        <div className={`glass-card p-6 border-4 ${
          message.type === 'success' 
            ? 'bg-green-50 text-green-800 border-green-400' 
            : 'bg-red-50 text-red-800 border-red-400'
        }`}>
          <p className="font-bold text-base">{message.text}</p>
          <button
            onClick={() => setMessage(null)}
            className="mt-3 text-base underline hover:no-underline font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Document List */}
      <div className="glass-card">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-bold text-primary-500">
              Uploaded Documents ({documents.length})
            </h3>
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-10 w-10 border-b-4 border-accent-pink mx-auto mb-6"></div>
              <p className="text-text-secondary text-lg font-medium">Loading documents...</p>
            </div>
          ) : documents.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="mx-auto h-16 w-16 text-primary-500 mb-6" />
              <p className="text-text-primary text-xl font-bold">No documents uploaded yet</p>
              <p className="text-base text-text-secondary mt-2 font-medium">
                Upload PDF files or web content to start asking questions
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="glass-card flex items-center justify-between p-6 hover:shadow-card-hover transition-all duration-300"
                >
                  <div className="flex items-center space-x-4 flex-1 min-w-0 overflow-hidden">
                    {doc.source_type === 'web' ? (
                      <Globe className="h-10 w-10 text-primary-500 flex-shrink-0" />
                    ) : (
                      <File className="h-10 w-10 text-accent-coral flex-shrink-0" />
                    )}
                    <div className="flex-1 min-w-0 overflow-hidden">
                      <div className="flex items-center space-x-2 min-w-0">
                        <h4 
                          className="text-base font-bold text-text-primary truncate max-w-full"
                          title={doc.name}
                          style={{
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                          }}
                        >
                          {doc.name}
                        </h4>
                        {doc.source_type === 'web' && doc.original_url && (
                          <a
                            href={doc.original_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-text-accent hover:text-primary-500 transition-colors flex-shrink-0"
                            title="View original webpage"
                          >
                            <ExternalLink className="h-4 w-4" />
                          </a>
                        )}
                      </div>
                      <p 
                        className="text-base text-text-secondary mt-2 line-clamp-2 font-medium"
                        title={doc.summary}
                        style={{
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                          wordWrap: 'break-word'
                        }}
                      >
                        {doc.summary}
                      </p>
                      <div className="flex items-center space-x-6 mt-3 text-sm text-text-muted font-medium">
                        <span className="flex items-center">
                          <Clock className="h-4 w-4 mr-2" />
                          {formatDate(doc.created_at)}
                        </span>
                        <span>{formatFileSize(doc.file_size)}</span>
                        <span>{doc.chunk_count} excerpts</span>
                        <span className={`px-2 py-1 rounded-card text-xs font-bold ${
                          doc.source_type === 'web' 
                            ? 'bg-accent-sage text-text-primary' 
                            : 'bg-primary-500 text-text-light'
                        }`}>
                          {doc.source_type === 'web' ? 'WEB' : 'PDF'}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.id, doc.name)}
                    className="ml-6 p-3 text-text-muted hover:text-accent-coral transition-all duration-300 rounded-card hover:bg-muted-hover hover:shadow-glass"
                    title="Delete document"
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentManager;