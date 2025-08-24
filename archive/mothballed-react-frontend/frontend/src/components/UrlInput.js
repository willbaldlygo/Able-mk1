/**
 * URL Input Component for Able3
 */
import React, { useState } from 'react';
import { Globe, Plus, AlertCircle } from 'lucide-react';

const UrlInput = ({ onUrlSubmit, loading, vintage = false }) => {
  const [url, setUrl] = useState('');
  const [title, setTitle] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const validateUrl = (urlString) => {
    try {
      const url = new URL(urlString);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (e) {
      return false;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    if (!validateUrl(url.trim())) {
      setError('Please enter a valid HTTP or HTTPS URL');
      return;
    }

    setError('');
    setIsSubmitting(true);

    try {
      await onUrlSubmit({
        url: url.trim(),
        title: title.trim() || undefined
      });
      
      // Clear form on success
      setUrl('');
      setTitle('');
    } catch (error) {
      setError(error.message || 'Failed to process URL');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
    if (error) setError(''); // Clear error when user starts typing
  };

  const handleTitleChange = (e) => {
    setTitle(e.target.value);
  };

  if (vintage) {
    return (
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="vintage-input-bar">
          <input
            type="url"
            value={url}
            onChange={handleUrlChange}
            placeholder="https://example.com/article"
            disabled={loading || isSubmitting}
            className="vintage-input-field"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
        </div>
        <div className="vintage-input-bar">
          <input
            type="text"
            value={title}
            onChange={handleTitleChange}
            placeholder="Override the page title..."
            disabled={loading || isSubmitting}
            className="vintage-input-field"
          />
        </div>
        <button
          type="submit"
          disabled={loading || isSubmitting || !url.trim()}
          className="vintage-send-btn w-full"
        >
          {isSubmitting ? (
            <div className="flex items-center justify-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
              <span>Processing...</span>
            </div>
          ) : (
            <span>Add Web Content</span>
          )}
        </button>
        {error && (
          <div className="text-red-600 text-sm vintage-text">
            {error}
          </div>
        )}
        {isSubmitting && (
          <div className="text-center vintage-text text-sm">
            Processing URL...
          </div>
        )}
      </form>
    );
  }

  return (
    <div className="glass-card p-6 mb-6">
      <div className="flex items-center space-x-3 mb-4">
        <Globe className="h-6 w-6 text-accent-pink" />
        <h3 className="text-lg font-bold text-text-primary gradient-text">Add Web Content</h3>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="url" className="block text-sm font-bold text-text-primary mb-2">
            Website URL *
          </label>
          <input
            type="url"
            id="url"
            value={url}
            onChange={handleUrlChange}
            placeholder="https://example.com/article"
            disabled={loading || isSubmitting}
            className="w-full p-3 border-4 border-text-primary rounded-card focus:ring-2 focus:ring-text-accent focus:border-text-accent disabled:bg-glass-light disabled:text-text-muted font-medium text-base"
          />
        </div>
        
        <div>
          <label htmlFor="title" className="block text-sm font-bold text-text-primary mb-2">
            Custom Title (optional)
          </label>
          <input
            type="text"
            id="title"
            value={title}
            onChange={handleTitleChange}
            placeholder="Override the page title..."
            disabled={loading || isSubmitting}
            className="w-full p-3 border-4 border-text-primary rounded-card focus:ring-2 focus:ring-text-accent focus:border-text-accent disabled:bg-glass-light disabled:text-text-muted font-medium text-base"
          />
        </div>

        {error && (
          <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-card border-2 border-red-300">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span className="text-sm font-medium">{error}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || isSubmitting || !url.trim()}
          className="w-full flex items-center justify-center space-x-2 px-6 py-3 gradient-button text-white rounded-card disabled:bg-text-muted disabled:cursor-not-allowed transition-all duration-300 shadow-card border-4 border-text-primary font-bold"
        >
          {isSubmitting ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Saving...</span>
            </>
          ) : (
            <>
              <Plus className="h-4 w-4" />
              <span>Add Web Content</span>
            </>
          )}
        </button>
      </form>

      <p className="text-sm text-text-secondary mt-3 font-medium">
        Enter any web article or page URL. Content will be saved as text files for offline research.
      </p>
    </div>
  );
};

export default UrlInput;