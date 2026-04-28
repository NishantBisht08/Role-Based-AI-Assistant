import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import API from "./api";
import { loadUserChats, addChatToHistory } from "./layout";

// Generate a unique chat session ID
function newChatId() {
  return "chat_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
}

const CSS = `
  .chat-wrap {
    flex: 1; display:flex;flex-direction:column;overflow:hidden;
    background: var(--surface);
  }
  .chat-msgs {
    flex:1;overflow-y:auto;padding:28px 24px;
    display:flex;flex-direction:column;gap:22px;
  }

  /* empty state */
  .chat-empty {
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;height:100%;gap:14px;text-align:center;
    padding:40px; animation: fadeUp 0.4s ease;
  }
  .ce-icon {
    width:58px;height:58px;border-radius:17px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 0 30px rgba(99,102,241,0.35);margin-bottom:4px;
  }
  .ce-icon svg { width:26px;height:26px;color:white; }
  .ce-title { font-size:22px;font-weight:700;color:var(--tx);letter-spacing:-0.4px; }
  .ce-sub { font-size:14px;color:var(--tx2);max-width:380px;line-height:1.7; }
  .chip-row { display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:6px; }
  .chip {
    padding:8px 14px;background:var(--surface2);
    border:1px solid var(--bd);border-radius:20px;
    font-size:13px;color:var(--tx2);cursor:pointer;
    transition:all 0.15s;font-family:inherit;
  }
  .chip:hover { background:rgba(99,102,241,0.1);border-color:rgba(99,102,241,0.3);color:#818cf8; }

  /* message rows */
  .msg-row { display:flex;gap:11px;align-items:flex-start;animation:fadeUp 0.28s ease; }
  .msg-row.user { flex-direction:row-reverse; align-self:flex-end; max-width:75%; }
  .msg-row.ai   { align-self:flex-start; max-width:82%; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

  .msg-av {
    width:30px;height:30px;border-radius:9px;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
    font-size:11px;font-weight:700;margin-top:2px;
  }
  .msg-av.ai-av {
    background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;
    box-shadow:0 0 12px rgba(99,102,241,0.3);
  }
  .msg-av.user-av {
    background:var(--surface2);border:1px solid var(--bd);color:var(--tx2);
  }
  .msg-body { flex:1;min-width:0; }
  .msg-sender {
    font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;
    margin-bottom:4px;color:var(--tx3);
  }
  .msg-row.ai .msg-sender { color:#818cf8; }

  .bubble {
    padding:11px 15px;border-radius:13px;font-size:14px;line-height:1.7;
    color:var(--tx);word-break:break-word;
  }
  .bubble.user-bubble {
    background:rgba(99,102,241,0.13);
    border:1px solid rgba(99,102,241,0.18);
    border-bottom-right-radius:3px;
  }
  .bubble.ai-bubble {
    background:var(--surface2);border:1px solid var(--bd);
    border-bottom-left-radius:3px;
  }
  .bubble.loading { display:flex;align-items:center;gap:5px; }

  .src-tag {
    margin-top:5px;font-size:11px;color:var(--tx3);
    display:flex;align-items:center;gap:4px;
  }

  /* typing dots */
  .dot {
    width:6px;height:6px;border-radius:50%;background:var(--tx2);
    animation:dotBounce 1.1s infinite ease-in-out;
  }
  .dot:nth-child(2){animation-delay:.18s}
  .dot:nth-child(3){animation-delay:.36s}
  @keyframes dotBounce{0%,80%,100%{transform:scale(.8);opacity:.5}40%{transform:scale(1.15);opacity:1}}

  /* input area */
  .chat-foot {
    padding:14px 24px 18px;border-top:1px solid var(--bd);
    background:var(--surface);
  }
  .input-box {
    display:flex;align-items:flex-end;gap:9px;
    background:var(--surface2);border:1px solid var(--bd2);
    border-radius:14px;padding:10px 10px 10px 15px;
    transition:border-color 0.18s,box-shadow 0.18s;
    max-width:800px;margin:0 auto;
  }
  .input-box:focus-within {
    border-color:rgba(99,102,241,0.5);
    box-shadow:0 0 0 3px rgba(99,102,241,0.12);
  }
  .chat-ta {
    flex:1;background:none;border:none;outline:none;
    font-family:inherit;font-size:14px;color:var(--tx);
    resize:none;min-height:22px;max-height:150px;
    overflow-y:auto;line-height:1.5;padding:2px 0;
  }
  .chat-ta::placeholder{color:var(--tx3);}
  .send-btn {
    width:35px;height:35px;border-radius:10px;flex-shrink:0;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    border:none;cursor:pointer;color:#fff;
    display:flex;align-items:center;justify-content:center;
    transition:opacity 0.18s,transform 0.15s,box-shadow 0.18s;
  }
  .send-btn:hover:not(:disabled){opacity:0.9;transform:scale(1.06);box-shadow:0 0 16px rgba(99,102,241,0.4);}
  .send-btn:disabled{opacity:0.35;cursor:not-allowed;}
  .foot-hint{text-align:center;font-size:11px;color:var(--tx3);margin-top:7px;}
`;

const SUGGESTIONS = [
  "What is the role of HR?",
  "Summarize Q3 financials",
  "Recruitment policy overview",
  "How to reduce HR costs?",
];

function shouldShowSource(questionText = "") {
  return /(source|document|doc|file|where did this come from|citation|reference)/i.test(questionText);
}

function normalizeAssistantText(text = "") {
  const cleaned = String(text)
    .replace(/\baccording to\b[:\s]*/gi, "")
    .replace(/\bbased on\b[:\s]*/gi, "")
    .replace(/\bfrom\s+[a-z0-9_.-]+\.(md|txt|pdf|docx?)\b/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();

  if (!cleaned) return "I can help with that. Could you share a little more detail?";
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

export default function Ask() {
  const location = useLocation();
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [isFirstTimeUser, setIsFirstTimeUser] = useState(false);
  const [chatId, setChatId] = useState(() => newChatId());
  const bottomRef = useRef(null);
  const taRef = useRef(null);
  const username = localStorage.getItem("username") || "You";
  const role = localStorage.getItem("role") || "member";
  const initials = username.slice(0, 2).toUpperCase();

  // If a sidebar chat was clicked, load its messages
  useEffect(() => {
    const incomingId = location.state?.chatId;
    const all = loadUserChats(username, role);
    setIsFirstTimeUser(all.length === 0);
    if (incomingId) {
      setChatLoading(true);
      const found = all.find(c => c.id === incomingId);
      if (found) {
        setChatId(found.id);
        setMsgs(found.messages || []);
      } else {
        setChatId(newChatId());
        setMsgs([]);
      }
      setTimeout(() => setChatLoading(false), 180);
    } else {
      // New conversation
      setChatId(newChatId());
      setMsgs([]);
    }
  }, [location.state?.chatId, username, role]); // eslint-disable-line

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, loading]);

  const autoResize = () => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "22px";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  };

  const send = async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;

    const userMsg = { role: "user", text: q };
    const updatedMsgs = [...msgs, userMsg];
    setMsgs(updatedMsgs);
    setInput("");
    if (taRef.current) taRef.current.style.height = "22px";
    setLoading(true);

    // Title = first question, truncated to 40 chars
    const isFirst = msgs.length === 0;
    const title = isFirst ? (q.length > 40 ? q.slice(0, 40) + "…" : q) : null;

    try {
      const token = localStorage.getItem("access_token");
      const res = await API.post("/ask", { question: q, token });
      const aiMsg = {
        role: "ai",
        text: normalizeAssistantText(res.data.answer || "No answer."),
        source: shouldShowSource(q) ? res.data.source : null
      };
      const finalMsgs = [...updatedMsgs, aiMsg];
      setMsgs(finalMsgs);
      // Save to this user's chat history
      addChatToHistory(username, role, chatId, title || msgs[0]?.text?.slice(0, 40) || "Conversation", finalMsgs);
      setIsFirstTimeUser(false);
    } catch {
      const errMsg = { role: "ai", text: "Something went wrong. Please try again.", isError: true };
      const finalMsgs = [...updatedMsgs, errMsg];
      setMsgs(finalMsgs);
      addChatToHistory(username, role, chatId, title || "Conversation", finalMsgs);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{CSS}</style>
      <div className="chat-wrap">
        <div className="chat-msgs">
          {chatLoading ? (
            <div className="chat-empty">
              <div className="ce-title">Loading conversation...</div>
              <div className="ce-sub">Fetching your recent messages.</div>
            </div>
          ) : msgs.length === 0 && !loading ? (
            <div className="chat-empty">
              <div className="ce-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                  <rect x="3" y="11" width="18" height="11" rx="2"/>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                  <circle cx="9" cy="16" r="1" fill="currentColor"/>
                  <circle cx="15" cy="16" r="1" fill="currentColor"/>
                </svg>
              </div>
              <div className="ce-title">{isFirstTimeUser ? "Start your first conversation" : "AI Chat Assistant"}</div>
              <div className="ce-sub">
                {isFirstTimeUser
                  ? "Try one of these prompts to begin. Your assistant will respond in a natural, role-aware way."
                  : `Ask anything related to your ${role} scope and I will help with a concise answer.`
                }
              </div>
              <div className="chip-row">
                {SUGGESTIONS.map(s => (
                  <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {msgs.map((m, i) => (
                <div key={i} className={`msg-row ${m.role}`}>
                  <div className={`msg-av ${m.role === "ai" ? "ai-av" : "user-av"}`}>
                    {m.role === "ai"
                      ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                      : initials
                    }
                  </div>
                  <div className="msg-body">
                    <div className="msg-sender">{m.role === "ai" ? "NexusAI" : username}</div>
                    <div className={`bubble ${m.role === "ai" ? "ai-bubble" : "user-bubble"}`}
                      style={m.isError ? { color: "#f87171" } : {}}>
                      {m.text}
                    </div>
                    {m.source && (
                      <div className="src-tag">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        Source: {m.source}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="msg-row ai">
                  <div className="msg-av ai-av">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                  </div>
                  <div className="msg-body">
                    <div className="msg-sender">NexusAI</div>
                    <div className="bubble ai-bubble loading">
                      <div className="dot"/><div className="dot"/><div className="dot"/>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="chat-foot">
          <div className="input-box">
            <textarea
              ref={taRef}
              className="chat-ta"
              placeholder="Ask anything about your documents…"
              value={input}
              rows={1}
              onChange={e => { setInput(e.target.value); autoResize(); }}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }}}
            />
            <button className="send-btn" onClick={() => send()} disabled={!input.trim() || loading}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
          </div>
          <p className="foot-hint">Enter to send · Shift+Enter for new line · Powered by RAG + LLM</p>
        </div>
      </div>
    </>
  );
}