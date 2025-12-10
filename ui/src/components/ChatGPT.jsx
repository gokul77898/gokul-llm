import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import { Send, Menu, Plus, Sun, Moon, Copy, RotateCcw, Check } from 'lucide-react';
import { queryAPI } from '../api';
import ResultCard from './ResultCard';

const ChatGPT = () => {
  // State
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [chatSessions, setChatSessions] = useState(() => {
    const saved = localStorage.getItem('chatSessions');
    return saved ? JSON.parse(saved) : [{ id: Date.now(), title: 'New Chat', messages: [] }];
  });
  const [currentSessionId, setCurrentSessionId] = useState(() => chatSessions[0].id);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const [copiedId, setCopiedId] = useState(null);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, typingMessageId]);

  // Save to localStorage
  useEffect(() => {
    localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  // Load current session messages
  useEffect(() => {
    const session = chatSessions.find(s => s.id === currentSessionId);
    setMessages(session?.messages || []);
  }, [currentSessionId, chatSessions]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    // Add typing indicator
    const typingId = Date.now() + 1;
    setTypingMessageId(typingId);

    try {
      const response = await queryAPI({
        query: input,
        model: 'auto',
        top_k: 5,
        task: 'qa'  // Default to QA, could be made dynamic
      });

      // Remove typing indicator
      setTypingMessageId(null);

      const assistantMessage = {
        id: Date.now() + 2,
        role: 'assistant',
        content: response.answer || 'No response generated',
        model: response.auto_model_used || response.model,
        confidence: response.confidence,
        sources: response.sources || [],
        retrieved_docs: response.retrieved_docs || 0,
        timestamp: new Date().toISOString(),
        raw: response,  // Pass raw backend response
        metadata: response.metadata
      };

      const newMessages = [...messages, userMessage, assistantMessage];
      setMessages(newMessages);

      // Update session
      setChatSessions(prev => prev.map(session =>
        session.id === currentSessionId
          ? { 
              ...session, 
              messages: newMessages,
              title: session.messages.length === 0 ? input.slice(0, 30) : session.title
            }
          : session
      ));
    } catch (error) {
      setTypingMessageId(null);
      const errorMessage = {
        id: Date.now() + 2,
        role: 'error',
        content: `Error: ${error.message || 'Failed to get response'}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const newChat = () => {
    const newSession = {
      id: Date.now(),
      title: 'New Chat',
      messages: []
    };
    setChatSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    setMessages([]);
  };

  const copyMessage = (content, id) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const regenerate = () => {
    // Find last user message and resend
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
    if (lastUserMsg) {
      setInput(lastUserMsg.content);
      handleSend();
    }
  };

  return (
    <div className={`flex h-screen ${darkMode ? 'dark bg-gray-900' : 'bg-white'}`}>
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -300 }}
            animate={{ x: 0 }}
            exit={{ x: -300 }}
            transition={{ type: 'spring', damping: 25 }}
            className={`w-64 border-r ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-50 border-gray-200'} flex flex-col`}
          >
            {/* New Chat Button */}
            <div className="p-3">
              <button
                onClick={newChat}
                className={`w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border ${
                  darkMode 
                    ? 'border-gray-600 hover:bg-gray-700 text-white' 
                    : 'border-gray-300 hover:bg-gray-100 text-gray-900'
                } transition-colors`}
              >
                <Plus size={18} />
                <span className="text-sm font-medium">New chat</span>
              </button>
            </div>

            {/* Chat Sessions */}
            <div className="flex-1 overflow-y-auto px-2">
              {chatSessions.map(session => (
                <button
                  key={session.id}
                  onClick={() => setCurrentSessionId(session.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg mb-1 text-sm truncate transition-colors ${
                    session.id === currentSessionId
                      ? darkMode ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-900'
                      : darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  {session.title}
                </button>
              ))}
            </div>

            {/* Dark Mode Toggle */}
            <div className="p-3 border-t border-gray-700">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg ${
                  darkMode ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-100 text-gray-900'
                }`}
              >
                {darkMode ? <Sun size={18} /> : <Moon size={18} />}
                <span className="text-sm">{darkMode ? 'Light' : 'Dark'} Mode</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className={`border-b ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} px-4 py-3 flex items-center gap-3`}>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
          >
            <Menu size={20} className={darkMode ? 'text-white' : 'text-gray-900'} />
          </button>
          <h1 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Vakeels.AI - Legal Assistant
          </h1>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <h2 className={`text-3xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  How can I help you today?
                </h2>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Ask me anything about legal matters
                </p>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto py-6">
              {messages.map(msg => (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  darkMode={darkMode}
                  onCopy={copyMessage}
                  copied={copiedId === msg.id}
                />
              ))}
              {typingMessageId && <TypingIndicator darkMode={darkMode} />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className={`border-t ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} p-4`}>
          <div className="max-w-3xl mx-auto">
            <div className={`flex items-end gap-2 p-3 rounded-2xl border ${
              darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-300'
            }`}>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Message Vakeels.AI..."
                rows={1}
                className={`flex-1 resize-none outline-none bg-transparent ${
                  darkMode ? 'text-white placeholder-gray-400' : 'text-gray-900 placeholder-gray-500'
                } max-h-40`}
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className={`p-2 rounded-lg transition-colors ${
                  input.trim() && !loading
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-300 text-gray-500'
                }`}
              >
                <Send size={20} />
              </button>
            </div>
            <p className={`text-xs text-center mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              Powered by AutoPipeline • ChromaDB • Model Selector
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Message Bubble Component
const MessageBubble = ({ message, darkMode, onCopy, copied }) => {
  const [showActions, setShowActions] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`group px-6 py-6 ${
        message.role === 'assistant'
          ? darkMode ? 'bg-gray-800' : 'bg-gray-50'
          : ''
      }`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className="max-w-3xl mx-auto">
        <div className="flex gap-4">
          {/* Avatar */}
          <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
            message.role === 'user'
              ? 'bg-blue-600 text-white'
              : message.role === 'error'
              ? 'bg-red-600 text-white'
              : 'bg-green-600 text-white'
          }`}>
            {message.role === 'user' ? 'U' : message.role === 'error' ? '!' : 'AI'}
          </div>

          {/* Content */}
          <div className="flex-1 space-y-2">
            {message.role === 'user' ? (
              <div className={`prose prose-sm max-w-none ${darkMode ? 'prose-invert' : ''}`}>
                <p className={darkMode ? 'text-white' : 'text-gray-900'}>{message.content}</p>
              </div>
            ) : message.role === 'assistant' && message.raw ? (
              <ResultCard result={message.raw} />
            ) : (
              <div className={`prose prose-sm max-w-none ${darkMode ? 'prose-invert' : ''}`}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={`${darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-200 text-gray-900'} px-1.5 py-0.5 rounded text-sm`} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            )}

            {/* Metadata */}
            {message.model && (
              <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'} flex items-center gap-3`}>
                <span>Model: {message.model}</span>
                {message.confidence && (
                  <span>Confidence: {(message.confidence * 100).toFixed(1)}%</span>
                )}
                {message.retrieved_docs !== undefined && (
                  <span>Sources: {message.retrieved_docs} docs</span>
                )}
              </div>
            )}

            {/* Actions */}
            <AnimatePresence>
              {showActions && message.role === 'assistant' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex gap-2"
                >
                  <button
                    onClick={() => onCopy(message.content, message.id)}
                    className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
                      darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-200 text-gray-600'
                    }`}
                  >
                    {copied ? <Check size={14} /> : <Copy size={14} />}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Typing Indicator
const TypingIndicator = ({ darkMode }) => (
  <div className={`px-6 py-6 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
    <div className="max-w-3xl mx-auto flex gap-4">
      <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center flex-shrink-0">
        AI
      </div>
      <div className="flex gap-1 items-center">
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
          className={`w-2 h-2 rounded-full ${darkMode ? 'bg-gray-400' : 'bg-gray-600'}`}
        />
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
          className={`w-2 h-2 rounded-full ${darkMode ? 'bg-gray-400' : 'bg-gray-600'}`}
        />
        <motion.div
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
          className={`w-2 h-2 rounded-full ${darkMode ? 'bg-gray-400' : 'bg-gray-600'}`}
        />
      </div>
    </div>
  </div>
);

export default ChatGPT;
