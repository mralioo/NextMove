import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, MicOff, Volume2, VolumeX, Copy, Check, ArrowUp, 
  Sparkles, Plus, Clock, ChevronRight, Bot, ThumbsUp, ThumbsDown
} from 'lucide-react';
import ThinkingPanel from '../components/ThinkingPanel';
import { useSpeech } from '../hooks/useSpeech';
import { api } from '../api/client';

const THINKING_STEPS = [
  { icon: 'brain',       step: 'Extracting intent and classifying query...' },
  { icon: 'fingerprint', step: 'Building HCADE situation fingerprint...' },
  { icon: 'tools',       step: 'Planning analytical tool execution...' },
  { icon: 'execute',     step: 'Executing analytics tools against dataset...' },
  { icon: 'history',     step: 'Searching historical records via HCADE similarity...' },
  { icon: 'llm',         step: 'Generating evidence-grounded response with gpt-5.6-luna...' },
];

// Helper to format assistant response with clean Markdown (bold, lists, headers) like ChatGPT
const renderFormattedContent = (content) => {
  if (!content) return null;
  const lines = content.split('\n');
  return lines.map((line, idx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return <div key={idx} style={{ height: '8px' }} />;
    }
    // Headings
    if (trimmed.startsWith('### ')) {
      return (
        <h4 key={idx} style={{ color: 'var(--teal)', fontSize: '0.95rem', margin: '10px 0 4px 0', fontWeight: 700 }}>
          {trimmed.replace('### ', '')}
        </h4>
      );
    }
    if (trimmed.startsWith('## ') || trimmed.startsWith('RECOMMENDATION:') || trimmed.startsWith('Situation summary:')) {
      return (
        <div key={idx} style={{ color: 'var(--gold)', fontWeight: 700, fontSize: '0.95rem', margin: '8px 0 4px 0' }}>
          {trimmed.replace('## ', '')}
        </div>
      );
    }
    // Bullet points
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      return (
        <div key={idx} style={{ display: 'flex', gap: '8px', marginLeft: '4px', margin: '3px 0' }}>
          <span style={{ color: 'var(--teal)', fontWeight: 'bold' }}>•</span>
          <span>{renderInlineBold(trimmed.substring(2))}</span>
        </div>
      );
    }
    // Numbered lists
    if (/^\d+\.\s/.test(trimmed)) {
      const match = trimmed.match(/^(\d+\.)\s(.*)/);
      return (
        <div key={idx} style={{ display: 'flex', gap: '8px', marginLeft: '4px', margin: '3px 0' }}>
          <span style={{ color: 'var(--teal)', fontWeight: '600' }}>{match[1]}</span>
          <span>{renderInlineBold(match[2])}</span>
        </div>
      );
    }
    // Regular paragraph
    return (
      <p key={idx} style={{ margin: '4px 0', lineHeight: 1.6 }}>
        {renderInlineBold(line)}
      </p>
    );
  });
};

// Bold parser for **text**
const renderInlineBold = (text) => {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} style={{ color: '#FFFFFF', fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
};

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [thinkingSteps, setThinkingSteps] = useState(THINKING_STEPS.map(s => ({ ...s, status: 'pending', detail: '' })));
  const [isProcessing, setIsProcessing] = useState(false);
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState(null);
  const [feedbackGiven, setFeedbackGiven] = useState({}); // { msgId: 'up'|'down'|'sent' }
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  // Track the last user query for attaching to feedback
  const lastQueryRef = useRef('');
  
  const { speak, stopSpeaking, isSpeaking, startListening, stopListening, isListening, transcript } = useSpeech();

  useEffect(() => {
    connectWs();
    return () => {
      // Cancel any pending reconnect so no new connections are created after unmount
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent onclose from scheduling another reconnect
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (isListening && transcript) {
      setInput(transcript);
    }
  }, [transcript, isListening]);

  const connectWs = () => {
    // Guard: don't open a second connection if one is already open or connecting
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:8000/ws/chat`;
    
    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'thinking') {
            setThinkingSteps(prev => {
              const next = [...prev];
              if (next[data.index]) {
                next[data.index] = { ...next[data.index], status: 'active' };
                if (data.index > 0 && next[data.index - 1].status === 'active') {
                  next[data.index - 1] = { ...next[data.index - 1], status: 'done' };
                }
              }
              return next;
            });
          } else if (data.type === 'thinking_done') {
            setThinkingSteps(prev => {
              const next = [...prev];
              if (next[data.index]) {
                next[data.index] = { ...next[data.index], status: 'done', detail: data.detail || '' };
              }
              return next;
            });
          } else if (data.type === 'chunk') {
            // As soon as the first chunk arrives, create or append to agent message
            setMessages(prev => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === 'agent' && last.isStreaming) {
                next[next.length - 1] = { ...last, content: last.content + data.text };
                return next;
              } else {
                // First token: create new agent message
                return [
                  ...next,
                  {
                    id: Date.now(),
                    role: 'agent',
                    content: data.text,
                    timestamp: new Date(),
                    isStreaming: true
                  }
                ];
              }
            });
          } else if (data.type === 'done') {
            setMessages(prev => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last && last.role === 'agent') {
                next[next.length - 1] = {
                  ...last,
                  isStreaming: false,
                  latency:     data.latency,
                  responseId:  data.response_id || '',
                  queryType:   data.query_type  || '',
                };
              }
              return next;
            });
            setIsProcessing(false);
          } else if (data.type === 'error') {
            setMessages(prev => [...prev, {
              id: Date.now(),
              role: 'agent',
              content: 'Error: ' + data.message,
              timestamp: new Date(),
              isStreaming: false
            }]);
            setIsProcessing(false);
          }
        } catch (e) {
          console.error('Error parsing WS message', e);
        }
      };

      wsRef.current.onclose = () => {
        reconnectTimerRef.current = setTimeout(connectWs, 2500);
      };
    } catch (e) {
      console.error('WebSocket connection error', e);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, thinkingSteps, isProcessing]);

  const sendMessage = (textToSend) => {
    const text = textToSend || input;
    if (!text.trim() || isProcessing) return;
    lastQueryRef.current = text;  // track for feedback
    
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: text,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsProcessing(true);
    setThinkingSteps(THINKING_STEPS.map(s => ({ ...s, status: 'pending', detail: '' })));

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ text }));
    } else {
      setTimeout(() => {
        setMessages(prev => [
          ...prev,
          {
            id: Date.now() + 1,
            role: 'agent',
            content: 'WebSocket server unavailable. Please ensure python backend/run.py is running on http://localhost:8000.',
            timestamp: new Date(),
            isStreaming: false
          }
        ]);
        setIsProcessing(false);
      }, 1000);
    }
  };

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleFeedback = async (msg, rating) => {
    if (feedbackGiven[msg.id]) return; // already rated
    setFeedbackGiven(prev => ({ ...prev, [msg.id]: rating > 0 ? 'up' : 'down' }));
    try {
      await api.submitFeedback(
        msg.responseId,
        rating,
        '',
        lastQueryRef.current,
        msg.queryType || ''
      );
    } catch (e) {
      console.warn('Feedback submission failed:', e);
    }
    // Show confirmation briefly
    setTimeout(() => {
      setFeedbackGiven(prev => ({ ...prev, [msg.id]: 'sent' }));
    }, 500);
  };

  const quickPills = [
    { label: 'U6 Suspension', query: 'Line U6 is suspended between Hallesches Tor and Kaiserin-Augusta-Strasse. How should we react and reroute?' },
    { label: 'GNR Concert', query: 'Guns N Roses concert on June 23rd at Uber Arena. What flow changes should we expect?' },
    { label: 'Energy Efficiency', query: 'Which metro line has the worst energy-per-passenger efficiency ratio and why?' },
    { label: 'Top 5 Fragile Stations', query: 'Rank the five stations whose closure would fragment the network the most.' },
    { label: 'Weather Peak', query: 'Give an example of a passenger flow peak caused by bad weather in July.' },
  ];

  // Active reasoning step description
  const activeThinkingStep = thinkingSteps.find(s => s.status === 'active')?.step || 'Analyzing situation...';

  // Has the agent started streaming text for the current user question?
  const isAgentGenerating = messages.length > 0 && messages[messages.length - 1].role === 'agent';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      
      {/* Subheader controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 4px 12px 4px',
        borderBottom: '1px solid var(--border-subtle)',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text)' }}>Conversation</span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Secure Session · Berlin Local Time</span>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setMessages([])}
            className="pill-tab"
            style={{ fontSize: '0.75rem', padding: '4px 14px', border: '1px solid var(--border-subtle)' }}
          >
            + New Chat
          </button>
        </div>
      </div>

      {/* Main Chat & Thinking Panel Area */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden', gap: '20px', paddingBottom: '110px' }}>
        
        {/* Messages Stream */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
          paddingRight: '6px'
        }}>
          {messages.length === 0 ? (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              gap: '16px',
              textAlign: 'center'
            }}>
              <div style={{
                width: '64px', height: '64px', borderRadius: '50%',
                background: 'rgba(0, 229, 212, 0.1)',
                border: '1px solid rgba(0, 229, 212, 0.3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: '0 0 24px rgba(0, 229, 212, 0.2)'
              }}>
                <Sparkles size={28} color="var(--teal)" />
              </div>
              <h2 style={{ fontSize: '1.4rem', margin: 0, color: 'var(--text)', fontWeight: 700 }}>
                Talk To My Train
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '420px', margin: 0 }}>
                Ask operations questions about passenger flows, cascade disruptions, energy efficiency, and concert surges.
              </p>

              {/* Starter Question Cards */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '10px',
                marginTop: '16px',
                maxWidth: '640px',
                width: '100%'
              }}>
                {quickPills.slice(0, 4).map(p => (
                  <button
                    key={p.label}
                    onClick={() => { setInput(p.query); sendMessage(p.query); }}
                    style={{
                      background: 'rgba(6, 22, 29, 0.7)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-md)',
                      padding: '12px 16px',
                      color: 'var(--text)',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--teal)'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-subtle)'}
                  >
                    <div>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--teal)', marginBottom: '2px' }}>
                        {p.label}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '240px' }}>
                        {p.query}
                      </div>
                    </div>
                    <ChevronRight size={14} color="var(--text-muted)" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                
                {/* Assistant header avatar bar */}
                {msg.role === 'agent' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', paddingLeft: '2px' }}>
                    <div style={{ width: '22px', height: '22px', borderRadius: '50%', background: 'rgba(0, 229, 212, 0.2)', border: '1px solid var(--teal)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Bot size={13} color="var(--teal)" />
                    </div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--teal)', letterSpacing: '0.02em' }}>Talk To My Train</span>
                  </div>
                )}

                {/* Bubble */}
                <div className={`msg-${msg.role} ${msg.isStreaming ? 'msg-streaming' : ''}`}>
                  {msg.role === 'agent' ? (
                    <div>{renderFormattedContent(msg.content)}</div>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  )}
                </div>

                {/* Agent Metadata & Controls Bar */}
                {msg.role === 'agent' && !msg.isStreaming && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px', paddingLeft: '4px' }}>
                    <button
                      className="pill-tab"
                      style={{ padding: '3px 8px', fontSize: '0.72rem', border: '1px solid var(--border-subtle)' }}
                      onClick={() => isSpeaking ? stopSpeaking() : speak(msg.content)}
                      title="Speak response"
                    >
                      {isSpeaking ? <VolumeX size={13} color="var(--danger)" /> : <Volume2 size={13} color="var(--teal)" />}
                      <span>{isSpeaking ? 'Stop' : 'Speak'}</span>
                    </button>

                    <button
                      className="pill-tab"
                      style={{ padding: '3px 8px', fontSize: '0.72rem', border: '1px solid var(--border-subtle)' }}
                      onClick={() => handleCopy(msg.id, msg.content)}
                      title="Copy response"
                    >
                      {copiedId === msg.id ? <Check size={13} color="var(--teal)" /> : <Copy size={13} />}
                      <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                    </button>

                    {/* Feature 2: Feedback buttons — only shown when response has an ID */}
                    {msg.responseId && (
                      feedbackGiven[msg.id] === 'sent' ? (
                        <span style={{ fontSize: '0.68rem', color: 'var(--teal)', opacity: 0.8 }}>Thanks for the feedback!</span>
                      ) : (
                        <>
                          <button
                            className="pill-tab"
                            title="This was helpful"
                            disabled={!!feedbackGiven[msg.id]}
                            style={{
                              padding: '3px 8px', fontSize: '0.72rem',
                              border: `1px solid ${feedbackGiven[msg.id] === 'up' ? 'var(--teal)' : 'var(--border-subtle)'}`,
                              opacity: feedbackGiven[msg.id] && feedbackGiven[msg.id] !== 'up' ? 0.3 : 1,
                            }}
                            onClick={() => handleFeedback(msg, 1)}
                          >
                            <ThumbsUp size={12} color={feedbackGiven[msg.id] === 'up' ? 'var(--teal)' : undefined} />
                          </button>
                          <button
                            className="pill-tab"
                            title="This was not helpful"
                            disabled={!!feedbackGiven[msg.id]}
                            style={{
                              padding: '3px 8px', fontSize: '0.72rem',
                              border: `1px solid ${feedbackGiven[msg.id] === 'down' ? 'var(--danger)' : 'var(--border-subtle)'}`,
                              opacity: feedbackGiven[msg.id] && feedbackGiven[msg.id] !== 'down' ? 0.3 : 1,
                            }}
                            onClick={() => handleFeedback(msg, -1)}
                          >
                            <ThumbsDown size={12} color={feedbackGiven[msg.id] === 'down' ? 'var(--danger)' : undefined} />
                          </button>
                        </>
                      )
                    )}

                    {msg.latency && (
                      <span className="pill-badge active" style={{ fontSize: '0.65rem', fontFamily: 'JetBrains Mono' }}>
                        <Clock size={11} /> {msg.latency}s
                      </span>
                    )}

                    {msg.content.includes("HIGH confidence") && (
                      <span className="pill-badge teal" style={{ fontSize: '0.65rem' }}>HIGH CONFIDENCE</span>
                    )}
                    {msg.content.includes("MEDIUM confidence") && (
                      <span className="pill-badge active" style={{ fontSize: '0.65rem' }}>MEDIUM CONFIDENCE</span>
                    )}
                  </div>
                )}

                {/* User Message Timestamp */}
                {msg.role === 'user' && (
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: '4px', paddingRight: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <span>{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    <span style={{ color: 'var(--teal)' }}>✓</span>
                  </div>
                )}
              </div>
            ))
          )}

          {/* Clean ChatGPT-style Thinking Indicator while model is executing tools and before text arrives */}
          {isProcessing && !isAgentGenerating && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingLeft: '2px' }}>
                <div style={{ width: '22px', height: '22px', borderRadius: '50%', background: 'rgba(0, 229, 212, 0.2)', border: '1px solid var(--teal)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Bot size={13} color="var(--teal)" />
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--teal)' }}>Talk To My Train</span>
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '10px 18px',
                background: 'rgba(6, 22, 29, 0.75)',
                border: '1px solid rgba(0, 229, 212, 0.22)',
                borderRadius: '16px 16px 16px 4px',
                boxShadow: '0 4px 16px rgba(0,0,0,0.35)'
              }}>
                <div style={{
                  width: '14px', height: '14px',
                  border: '2px solid var(--teal)', borderTopColor: 'transparent',
                  borderRadius: '50%', animation: 'spin 0.8s linear infinite'
                }} />
                <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)', fontFamily: 'Inter, sans-serif' }}>
                  {activeThinkingStep}
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
        
        {/* Thinking Steps Panel (Right side of chat) */}
        {isProcessing && (
          <ThinkingPanel steps={thinkingSteps} visible={isProcessing} />
        )}
      </div>

      {/* Floating Capsule Chat Input Bar */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        background: 'linear-gradient(to top, rgba(2,11,14,0.98) 60%, rgba(2,11,14,0) 100%)',
        paddingTop: '20px'
      }}>
        {/* Input Capsule Bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          background: 'rgba(5, 20, 26, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(0, 229, 212, 0.22)',
          borderRadius: 'var(--radius-pill)',
          padding: '8px 12px 8px 16px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.55)'
        }}>
          {/* Quick Plus Icon */}
          <button
            onClick={() => setInput(quickPills[0].query)}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '4px'
            }}
            title="Insert scenario"
          >
            <Plus size={18} />
          </button>

          {/* Voice Input Button */}
          <button
            onClick={() => isListening ? stopListening() : startListening((res) => { setInput(res); sendMessage(res); })}
            style={{
              background: isListening ? 'rgba(255, 92, 92, 0.2)' : 'transparent',
              border: isListening ? '1px solid var(--danger)' : 'none',
              borderRadius: '50%',
              padding: '6px',
              color: isListening ? 'var(--danger)' : 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
            title="Speech to Text"
          >
            {isListening ? <MicOff size={18} /> : <Mic size={18} />}
          </button>

          {/* Text Input */}
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage(input)}
            placeholder={isListening ? "Listening... speak now..." : "Message Talk To My Train (e.g. U6 closure, energy ratio, concert surge)..."}
            disabled={isProcessing}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              color: 'var(--text)',
              fontSize: '0.92rem',
              outline: 'none',
              fontFamily: 'Inter, sans-serif'
            }}
          />

          {/* Send Button */}
          <button
            onClick={() => sendMessage(input)}
            disabled={isProcessing || !input.trim()}
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '50%',
              background: input.trim() && !isProcessing ? '#FFFFFF' : 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              color: input.trim() && !isProcessing ? '#020B0E' : 'var(--text-dim)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: input.trim() && !isProcessing ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              boxShadow: input.trim() && !isProcessing ? '0 0 14px rgba(255, 255, 255, 0.3)' : 'none'
            }}
          >
            <ArrowUp size={19} strokeWidth={2.5} />
          </button>
        </div>

        {/* Quick Suggestion Chips Below Input */}
        <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', padding: '0 8px 4px 8px', scrollbarWidth: 'none' }}>
          {quickPills.map(p => (
            <button
              key={p.label}
              onClick={() => { setInput(p.query); sendMessage(p.query); }}
              className="pill-tab"
              style={{
                fontSize: '0.72rem',
                padding: '4px 12px',
                background: 'rgba(5, 20, 26, 0.7)',
                border: '1px solid var(--border-subtle)',
                whiteSpace: 'nowrap'
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

    </div>
  );
}
