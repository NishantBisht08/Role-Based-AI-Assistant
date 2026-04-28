import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { loadUserChats } from "./layout";

const CSS = `
  .dash-page {
    flex: 1;
    overflow-y: auto;
    padding: 36px 32px;
    background: var(--surface);
  }
  .dash-greeting {
    font-size: 30px; font-weight: 700;
    color: var(--tx); letter-spacing: -0.6px;
    margin-bottom: 5px;
  }
  .dash-greeting span { color: #818cf8; }
  .dash-sub { font-size: 14px; color: var(--tx2); margin-bottom: 36px; line-height: 1.5; }
  .dash-sub strong { color: var(--tx); text-transform: capitalize; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }
  @media (max-width: 700px) { .stats-grid { grid-template-columns: 1fr; } }

  .stat-card {
    background: var(--surface2);
    border: 1px solid var(--bd);
    border-radius: 16px;
    padding: 22px 20px;
    transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
    cursor: pointer;
  }
  .stat-card:hover {
    border-color: var(--bd2);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
  }
  .stat-card:focus-visible {
    outline: 2px solid #818cf8;
    outline-offset: 2px;
  }
  .stat-icon {
    width: 38px; height: 38px; border-radius: 10px;
    display:flex;align-items:center;justify-content:center;
    margin-bottom: 14px;
  }
  .stat-icon.purple { background: rgba(99,102,241,0.12); color:#818cf8; }
  .stat-icon.green  { background: rgba(34,197,94,0.1); color:#4ade80; }
  .stat-icon.amber  { background: rgba(245,158,11,0.1); color:#fbbf24; }
  .stat-icon svg { width:18px;height:18px; }
  .stat-value { font-size:28px;font-weight:700;color:var(--tx);letter-spacing:-0.5px;margin-bottom:3px; }
  .stat-label { font-size:13px;color:var(--tx2); }

  .cta-card {
    background: var(--surface2);
    border: 1px solid var(--bd);
    border-radius: 18px;
    padding: 26px 24px;
    display: flex; align-items: center; gap: 20px;
    transition: border-color 0.18s, box-shadow 0.18s;
  }
  .cta-card:hover {
    border-color: rgba(99,102,241,0.3);
    box-shadow: 0 0 40px rgba(99,102,241,0.07);
  }
  .cta-icon-box {
    width: 52px; height: 52px; border-radius: 15px; flex-shrink:0;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    box-shadow: 0 0 22px rgba(99,102,241,0.35);
  }
  .cta-icon-box svg { width:24px;height:24px;color:white; }
  .cta-text { flex:1; }
  .cta-text h3 { font-size:16px;font-weight:700;color:var(--tx);margin-bottom:4px;letter-spacing:-0.2px; }
  .cta-text p { font-size:13px;color:var(--tx2);line-height:1.6; }
  .cta-go-btn {
    padding: 11px 22px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border:none;border-radius:11px;
    color:#fff;font-size:13px;font-weight:600;
    cursor:pointer;white-space:nowrap;flex-shrink:0;
    font-family:inherit;
    transition: opacity 0.18s, transform 0.15s, box-shadow 0.18s;
    box-shadow: 0 0 0 0 rgba(99,102,241,0);
  }
  .cta-go-btn:hover {
    opacity:0.9;transform:translateY(-1px);
    box-shadow: 0 6px 24px rgba(99,102,241,0.4);
  }
  .panel {
    margin-top: 16px;
    margin-bottom: 24px;
    background: var(--surface2);
    border: 1px solid var(--bd);
    border-radius: 14px;
    padding: 16px;
  }
  .panel-title { font-size: 14px; font-weight: 700; color: var(--tx); margin-bottom: 6px; }
  .panel-sub { font-size: 13px; color: var(--tx2); margin-bottom: 10px; }
  .panel-row { padding: 10px 0; border-top: 1px solid var(--bd); font-size: 13px; color: var(--tx2); }
  .panel-row:first-of-type { border-top: none; }
  .panel-row strong { color: var(--tx); margin-right: 6px; }
  @media(max-width:600px){.cta-card{flex-direction:column;align-items:flex-start;} .cta-go-btn{width:100%;}}
`;

export default function Dashboard() {
  const navigate = useNavigate();
  const username = localStorage.getItem("username") || "User";
  const role     = localStorage.getItem("role") || "member";
  const [activePanel, setActivePanel] = useState(null);
  const [panelLoading, setPanelLoading] = useState(false);
  const h = new Date().getHours();
  const greeting = h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
  const chats = useMemo(() => loadUserChats(username, role), [username, role]);
  const recentCount = chats.length;
  const docsFromChats = useMemo(() => {
    const docs = chats
      .flatMap((chat) => (chat.messages || []).map((m) => m.source).filter(Boolean))
      .filter(Boolean);
    return [...new Set(docs)].slice(0, 12);
  }, [chats]);

  const docPreviewByRole = {
    engineering: ["Architecture guidelines", "Deployment standards", "Service reliability checklist"],
    hr: ["Leave policy", "Hiring process guide", "Employee handbook"],
    finance: ["Expense policy", "Budget planning SOP", "Quarterly reporting notes"],
    member: ["Team collaboration policy", "Company onboarding guide", "General operations handbook"]
  };

  const permissionsByRole = {
    engineering: ["View engineering knowledge base", "Ask technical architecture questions", "Access deployment/process docs"],
    hr: ["View people operations policies", "Ask compliance and hiring questions", "Access employee procedure docs"],
    finance: ["View budgeting and reporting docs", "Ask finance process questions", "Access audit readiness material"],
    member: ["View shared company docs", "Ask role-safe assistant questions", "Read common policy content"]
  };

  const openPanel = (panelName) => {
    setPanelLoading(true);
    setActivePanel(panelName);
    setTimeout(() => setPanelLoading(false), 180);
  };

  const openChat = () => {
    if (chats.length > 0) {
      navigate("/ask", { state: { chatId: chats[0].id } });
      return;
    }
    navigate("/ask", { state: { mode: "new" } });
  };

  return (
    <>
      <style>{CSS}</style>
      <div className="dash-page">
        <h1 className="dash-greeting">{greeting}, <span>{username}</span> 👋</h1>
        <p className="dash-sub">
          Signed in as <strong>{role}</strong> · Your AI assistant is ready
        </p>

        {/* Stats */}
        <div className="stats-grid">
          <div className="stat-card" role="button" tabIndex={0} onClick={openChat} onKeyDown={(e) => e.key === "Enter" && openChat()}>
            <div className="stat-icon purple">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <div className="stat-value">{recentCount}</div>
            <div className="stat-label">Recent conversations</div>
          </div>
          <div className="stat-card" role="button" tabIndex={0} onClick={() => openPanel("documents")} onKeyDown={(e) => e.key === "Enter" && openPanel("documents")}>
            <div className="stat-icon green">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
            </div>
            <div className="stat-value">{docsFromChats.length || 3}</div>
            <div className="stat-label">Indexed documents</div>
          </div>
          <div className="stat-card" role="button" tabIndex={0} onClick={() => openPanel("role")} onKeyDown={(e) => e.key === "Enter" && openPanel("role")}>
            <div className="stat-icon amber">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <div className="stat-value" style={{ textTransform: "capitalize", fontSize: 20 }}>{role}</div>
            <div className="stat-label">Access level</div>
          </div>
        </div>

        {activePanel && (
          <div className="panel">
            {panelLoading ? (
              <>
                <div className="panel-title">Loading...</div>
                <div className="panel-sub">Preparing details.</div>
              </>
            ) : activePanel === "documents" ? (
              <>
                <div className="panel-title">Indexed Documents</div>
                <div className="panel-sub">Preview of available documents in your role scope.</div>
                {(docsFromChats.length > 0 ? docsFromChats : (docPreviewByRole[role] || docPreviewByRole.member)).map((doc) => (
                  <div className="panel-row" key={doc}>
                    <strong>Preview:</strong>{doc}
                  </div>
                ))}
              </>
            ) : (
              <>
                <div className="panel-title">Access Level</div>
                <div className="panel-sub">You are currently signed in as {role}.</div>
                {(permissionsByRole[role] || permissionsByRole.member).map((permission) => (
                  <div className="panel-row" key={permission}>
                    <strong>Permission:</strong>{permission}
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        {/* CTA */}
        <div className="cta-card">
          <div className="cta-icon-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="11" width="18" height="11" rx="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
              <circle cx="9" cy="16" r="1" fill="currentColor"/>
              <circle cx="15" cy="16" r="1" fill="currentColor"/>
            </svg>
          </div>
          <div className="cta-text">
            <h3>Start chatting with AI Assistant</h3>
            <p>Ask anything about your role-specific documents. Retrieves relevant context and generates precise answers instantly.</p>
          </div>
          <button className="cta-go-btn" onClick={openChat}>
            Open Chat →
          </button>
        </div>
      </div>
    </>
  );
}