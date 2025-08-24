/**
 * Enhanced Chat Interface for Able3
 */
import React, { useState, useEffect, useRef } from 'react';
import { Send, MessageCircle, Bot, User, ExternalLink, Mic, MicOff } from 'lucide-react';
import { apiService } from '../services/api';

const ChatInterface = ({ hasDocuments, vintage = false, currentModel, onModelError }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [responseStartTime, setResponseStartTime] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [accumulatedTranscript, setAccumulatedTranscript] = useState('');
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const silenceTimerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
    };
  }, []);



  const startSpeechRecognition = () => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = true; // Keep listening continuously
      recognition.interimResults = true; // Get interim results
      recognition.lang = 'en-US';

      recognitionRef.current = recognition;

      recognition.onstart = () => {
        setIsRecording(true);
        setIsTranscribing(false);
        // Clear any existing silence timer
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
        }
      };

      recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        // Process all results
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        // If we have a final result, add it to accumulated transcript
        if (finalTranscript) {
          const newAccumulated = accumulatedTranscript + (accumulatedTranscript ? ' ' : '') + finalTranscript;
          setAccumulatedTranscript(newAccumulated);
          setInputValue(newAccumulated + (interimTranscript ? ' ' + interimTranscript : ''));
          
          // Reset silence timer when we get speech
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
          }
          
          // Start new silence timer for 30 seconds
          silenceTimerRef.current = setTimeout(() => {
            stopSpeechRecognition();
          }, 30000);
        } else if (interimTranscript) {
          // Show interim results
          setInputValue(accumulatedTranscript + (accumulatedTranscript ? ' ' : '') + interimTranscript);
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        
        if (event.error === 'no-speech') {
          // Restart recognition after no-speech error
          setTimeout(() => {
            if (isRecording && recognitionRef.current) {
              recognition.start();
            }
          }, 100);
        } else if (event.error !== 'aborted') {
          setIsRecording(false);
          recognitionRef.current = null;
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
          }
          alert('Speech recognition failed. Please try again or type your question.');
        }
      };

      recognition.onend = () => {
        // Auto-restart if we're still supposed to be recording
        if (isRecording && recognitionRef.current) {
          setTimeout(() => {
            if (isRecording && recognitionRef.current) {
              recognition.start();
            }
          }, 100);
        }
      };

      recognition.start();
    } else {
      alert('Speech recognition is not supported in this browser. Please type your question.');
      setIsRecording(false);
    }
  };

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    setIsRecording(false);
    // Keep the accumulated transcript in the input
    if (accumulatedTranscript) {
      setInputValue(accumulatedTranscript);
    }
  };


  const toggleRecording = () => {
    if (isRecording) {
      stopSpeechRecognition();
    } else {
      // Reset accumulated transcript when starting fresh
      setAccumulatedTranscript('');
      startSpeechRecognition();
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
      model: currentModel?.name || 'Unknown'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setAccumulatedTranscript(''); // Reset accumulated transcript
    setLoading(true);
    setResponseStartTime(Date.now());

    try {
      const response = await apiService.chatWithDocuments(userMessage.content);
      const responseTime = Date.now() - responseStartTime;
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date(response.timestamp),
        model: currentModel?.name || 'Unknown',
        responseTime,
        modelType: currentModel?.type || 'unknown'
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const responseTime = Date.now() - responseStartTime;
      
      // Notify parent about model error
      if (error.response?.status === 503 && currentModel) {
        onModelError?.(currentModel, error);
      }
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: error.response?.data?.detail || 'I apologize, but I encountered an error while processing your question. Please try again.',
        timestamp: new Date(),
        error: true,
        model: currentModel?.name || 'Unknown',
        responseTime,
        modelError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setResponseStartTime(null);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatResponseTime = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const getModelTypeIcon = (type) => {
    switch (type) {
      case 'local':
        return '💻';
      case 'cloud':
        return '☁️';
      default:
        return '🤖';
    }
  };

  const getUniqueDocuments = (sources) => {
    const docs = new Map();
    sources.forEach(source => {
      if (!docs.has(source.document_id)) {
        docs.set(source.document_id, source.document_name);
      }
    });
    return Array.from(docs.values());
  };

  if (vintage) {
    return (
      <div className="flex flex-col h-full">
        {/* Expanded Messages Area */}
        <div className="flex-1 overflow-y-auto p-3 space-y-3 mb-3" style={{minHeight: 0}}>
          {messages.length === 0 ? (
            <div className="text-center h-full flex flex-col justify-center items-center">
              <svg width="48" height="48" viewBox="0 0 80 80" fill="none" className="mb-4 opacity-60">
                <rect x="20" y="30" width="40" height="30" rx="6" fill="#b9a67d" stroke="#4a5d4a" strokeWidth="2"/>
                <rect x="25" y="15" width="30" height="20" rx="4" fill="#c9b896" stroke="#4a5d4a" strokeWidth="2"/>
                <circle cx="32" cy="25" r="2" fill="#1e3a5f"/>
                <circle cx="48" cy="25" r="2" fill="#1e3a5f"/>
                <line x1="40" y1="15" x2="40" y2="8" stroke="#8b3a3a" strokeWidth="2" strokeLinecap="round"/>
                <circle cx="40" cy="5" r="2" fill="#d4834a" stroke="#8b3a3a" strokeWidth="1"/>
                <rect x="30" y="40" width="20" height="15" rx="2" fill="#e8dcc0" stroke="#4a5d4a" strokeWidth="1"/>
                <circle cx="35" cy="47" r="1.5" fill="#8b3a3a"/>
                <circle cx="45" cy="47" r="1.5" fill="#4a6b4a"/>
              </svg>
              <p className="vintage-text text-sm opacity-60">
                {hasDocuments ? 'Ready to answer your questions...' : 'Upload documents to start chatting'}
              </p>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div key={message.id} className="vintage-chat-message">
                  <div className={`vintage-message-bubble ${message.type === 'user' ? 'vintage-user-message' : 'vintage-bot-message'}`}>
                    <div className="vintage-text text-sm leading-relaxed" style={{whiteSpace: 'pre-wrap'}}>
                      {message.content}
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-2 border-t border-current opacity-75">
                        <p className="text-xs font-bold mb-2">
                          Sources ({getUniqueDocuments(message.sources).length}):
                        </p>
                        <div className="space-y-1">
                          {getUniqueDocuments(message.sources).slice(0, 3).map((docName, index) => (
                            <div key={index} className="text-xs opacity-80 truncate">
                              • {docName}
                            </div>
                          ))}
                          {getUniqueDocuments(message.sources).length > 3 && (
                            <div className="text-xs opacity-60">
                              +{getUniqueDocuments(message.sources).length - 3} more
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    <div className="text-xs opacity-50 mt-2 flex items-center justify-between">
                      <span>{formatTime(message.timestamp)}</span>
                      <div className="flex items-center space-x-2">
                        {message.model && (
                          <span className="flex items-center space-x-1">
                            <span>{getModelTypeIcon(message.modelType)}</span>
                            <span>{message.model}</span>
                          </span>
                        )}
                        {message.responseTime && (
                          <span className={`px-1 py-0.5 rounded text-xs ${
                            message.responseTime < 1000 ? 'bg-green-200 text-green-800' :
                            message.responseTime < 3000 ? 'bg-yellow-200 text-yellow-800' :
                            'bg-red-200 text-red-800'
                          }`}>
                            {formatResponseTime(message.responseTime)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="vintage-chat-message">
                  <div className="vintage-message-bubble vintage-bot-message">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      <span className="text-xs vintage-text ml-2">
                        {currentModel ? `${currentModel.name} thinking...` : 'Thinking...'}
                      </span>
                      {currentModel?.responseTime && (
                        <span className="text-xs opacity-60">
                          (~{formatResponseTime(currentModel.responseTime)})
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Panel - Moved to Bottom */}
        <div className="vintage-input-panel flex-shrink-0">
          <div className="vintage-input-section">
            <div className="vintage-input-bar">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={hasDocuments ? "Ask a question about your documents..." : "Upload documents first..."}
                disabled={!hasDocuments || loading || isRecording}
                className="vintage-input-field"
                rows={2}
              />
            </div>
          </div>
          <div 
            className={`vintage-mic-box ${isRecording ? 'vintage-mic-active' : ''}`}
            onClick={toggleRecording}
            title={isRecording ? "Stop recording" : "Start voice recording"}
          >
            <svg className="microphone" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" style={{color: '#4a5d4a'}}>
              <path d="M12 1a4 4 0 0 1 4 4v7a4 4 0 1 1-8 0V5a4 4 0 0 1 4-4z"/>
              <path d="M5 10a7 7 0 0 0 14 0"/>
              <line x1="12" y1="17" x2="12" y2="21"/>
              <line x1="8" y1="21" x2="16" y2="21"/>
            </svg>
          </div>
          <button
            onClick={sendMessage}
            disabled={!hasDocuments || !inputValue.trim() || loading || isRecording}
            className="vintage-send-btn"
            title="Send message"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b-2 border-accent-sage flex-shrink-0">
        <div className="flex items-center space-x-4">
          <MessageCircle className="h-7 w-7 text-primary-500" />
          <h2 className="text-2xl font-bold text-text-primary">
            Ask Questions
          </h2>
        </div>
        <p className="text-base text-text-secondary mt-2 font-medium">
          {hasDocuments 
            ? 'Ask questions about your uploaded documents'
            : 'Upload documents to start asking questions'
          }
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6" style={{minHeight: 0}}>
        {messages.length === 0 ? (
          <div className="text-center text-text-muted mt-12">
            <Bot className="mx-auto h-16 w-16 text-primary-500 mb-6" />
            <p className="text-xl font-bold text-text-primary">Ready to help with your research</p>
            <p className="text-base mt-3 font-medium">
              {hasDocuments 
                ? 'Ask me anything about your documents'
                : 'Upload some documents first'
              }
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl ${
                  message.type === 'user'
                    ? 'bg-primary-500 text-text-light border-2 border-transparent'
                    : message.error
                    ? 'bg-red-50 text-red-800 border-2 border-red-300'
                    : 'bg-accent-beige border-2 border-accent-sage text-text-primary'
                } rounded-card p-6 shadow-card`}
              >
                <div className="flex items-start space-x-3">
                  {message.type === 'bot' && (
                    <Bot className="h-6 w-6 mt-1 flex-shrink-0 text-primary-500" />
                  )}
                  {message.type === 'user' && (
                    <User className="h-6 w-6 mt-1 flex-shrink-0 text-text-light" />
                  )}
                  <div className="flex-1">
                    <p className="whitespace-pre-wrap text-base leading-relaxed font-medium">{message.content}</p>
                    
                    {/* Sources */}
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-6 pt-4 border-t-2 border-accent-sage">
                        <p className="text-base font-bold text-text-primary mb-3">
                          Sources ({getUniqueDocuments(message.sources).length} document{getUniqueDocuments(message.sources).length !== 1 ? 's' : ''}):
                        </p>
                        <div className="space-y-2">
                          {getUniqueDocuments(message.sources).map((docName, index) => (
                            <div 
                              key={index}
                              className="flex items-center text-base text-text-secondary font-medium"
                            >
                              <ExternalLink className="h-4 w-4 mr-2 text-primary-500" />
                              <span className="truncate">{docName}</span>
                            </div>
                          ))}
                        </div>
                        <div className="mt-4">
                          <details className="text-base">
                            <summary className="cursor-pointer text-text-secondary hover:text-text-primary font-bold">
                              View excerpts ({message.sources.length})
                            </summary>
                            <div className="mt-3 space-y-3">
                              {message.sources.map((source, index) => (
                                <div key={index} className="p-4 source-card rounded-card shadow-sm">
                                  <div className="font-bold text-text-primary mb-2">
                                    {source.document_name} (relevance: {(source.relevance_score * 100).toFixed(0)}%)
                                  </div>
                                  <div className="text-text-secondary text-sm leading-relaxed line-clamp-3">
                                    {source.chunk_content}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </details>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex items-center justify-between mt-4 text-sm opacity-75 font-medium">
                      <span>{formatTime(message.timestamp)}</span>
                      <div className="flex items-center space-x-2">
                        {message.model && (
                          <span className="flex items-center space-x-1">
                            <span>{getModelTypeIcon(message.modelType)}</span>
                            <span>{message.model}</span>
                          </span>
                        )}
                        {message.responseTime && (
                          <span className={`px-2 py-1 rounded text-xs ${
                            message.responseTime < 1000 ? 'bg-green-100 text-green-800' :
                            message.responseTime < 3000 ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {formatResponseTime(message.responseTime)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-accent-beige border-2 border-accent-sage rounded-card p-6 shadow-card">
              <div className="flex items-center space-x-3">
                <Bot className="h-6 w-6 text-primary-500" />
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce"></div>
                  <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
                <span className="text-base text-text-primary">
                  {currentModel ? `${currentModel.name} thinking...` : 'Thinking...'}
                </span>
                {currentModel?.responseTime && (
                  <span className="text-sm text-text-secondary">
                    (~{formatResponseTime(currentModel.responseTime)})
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-6 border-t-2 border-accent-sage flex-shrink-0">
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={hasDocuments ? "Ask a question about your documents..." : "Upload documents first..."}
              disabled={!hasDocuments || loading || isRecording}
              className="w-full min-h-[48px] max-h-32 p-4 pr-16 border-2 border-accent-sage rounded-card focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none disabled:bg-glass-light disabled:text-text-muted font-medium text-base bg-glass-white text-text-primary"
              rows={1}
            />
            {isTranscribing && (
              <div className="absolute inset-0 glass-overlay rounded-card flex items-center justify-center">
                <div className="flex items-center space-x-2 text-primary-500">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500"></div>
                  <span className="text-sm font-medium">Transcribing...</span>
                </div>
              </div>
            )}
          </div>
          
          {/* Voice Input Button */}
          <button
            onClick={toggleRecording}
            disabled={!hasDocuments || loading || isTranscribing}
            className={`min-h-[48px] px-4 py-4 rounded-card transition-all duration-300 shadow-card border-2 border-accent-sage ${
              isRecording
                ? 'bg-accent-coral text-text-light border-accent-coral hover:bg-red-600 animate-pulse'
                : 'bg-glass-white text-primary-500 hover:bg-accent-beige'
            } disabled:bg-text-muted disabled:cursor-not-allowed`}
            title={isRecording ? "Stop recording" : "Start voice recording"}
          >
            {isRecording ? (
              <MicOff className="h-5 w-5" />
            ) : (
              <Mic className="h-5 w-5" />
            )}
          </button>
          
          {/* Send Button */}
          <button
            onClick={sendMessage}
            disabled={!hasDocuments || !inputValue.trim() || loading || isRecording || isTranscribing}
            className={`min-h-[48px] px-4 py-4 rounded-card transition-all duration-300 shadow-card border-2 border-accent-sage ${
              (!hasDocuments || !inputValue.trim() || loading || isRecording || isTranscribing)
                ? 'bg-text-muted text-text-light cursor-not-allowed'
                : 'bg-primary-500 text-text-light hover:bg-accent-coral'
            }`}
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
        {hasDocuments && (
          <div className="mt-3 flex items-center justify-between">
            <p className="text-sm text-text-light opacity-60 font-medium">
              Press Enter to send, Shift+Enter for new line
            </p>
            {(isRecording || isTranscribing) && (
              <p className="text-sm text-primary-500 font-medium">
                {isRecording ? "🎙️ Listening... Stops after 30s of silence" : "🔄 Transcribing audio..."}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;