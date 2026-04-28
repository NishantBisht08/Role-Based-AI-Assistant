import React, { useState, useEffect, useCallback } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { useTheme } from "./App";

// ── helpers ──────────────────────────────────────────────────
export function getChatStorageKey(username, role) {
  return `nexusai_chats_${username}_${role || "member"}`;
}

export function loadUserChats(username, role) {
  if (!username) return [];
  try {
    return JSON.parse(localStorage.getItem(getChatStorageKey(username, role)) || "[]");
  } catch { return []; }
}

export function saveUserChats(username, role, chats) {
  if (!username) return;
  localStorage.setItem(getChatStorageKey(username, role), JSON.stringify(chats));
}

// Called from Ask.js after first message is sent
export function addChatToHistory(username, role, chatId, title, messages) {
  const existing = loadUserChats(username, role);
  const filtered = existing.filter(c => c.id !== chatId);
  const updated  = [{ id: chatId, title, messages, updatedAt: Date.now() }, ...filtered].slice(0, 5);
  saveUserChats(username, role, updated);
  // fire a storage event so Layout re-renders
  window.dispatchEvent(new Event("nexusai_chats_updated"));
}

const CSS = `
  .layout-root {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── sidebar ── */
  .sidebar {
    width: 258px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    border-right: 1px solid var(--bd);
    background: var(--sidebar-bg);
    transition: width 0.22s cubic-bezier(0.4,0,0.2,1);
  }
  .sidebar.closed { width: 0; border-right: none; }

  .sb-head {
    padding: 18px 16px 14px;
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid var(--bd);
    flex-shrink: 0;
  }
  .sb-logo {
    width: 32px; height: 32px; border-radius: 9px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    flex-shrink:0;
    box-shadow: 0 0 14px rgba(99,102,241,0.35);
  }
  .sb-brand { font-size:15px;font-weight:700;color:var(--tx);white-space:nowrap;letter-spacing:-0.2px; }
  .sb-brand span { color:#818cf8; }

  .sb-new-btn {
    margin: 12px 12px 6px;
    padding: 9px 12px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    color: #818cf8;
    font-size: 13px; font-weight: 500;
    cursor: pointer;
    display: flex; align-items: center; gap: 8px;
    transition: background 0.18s, border-color 0.18s;
    white-space: nowrap;
    font-family: inherit;
  }
  .sb-new-btn:hover { background: rgba(99,102,241,0.18); border-color: rgba(99,102,241,0.35); }

  .sb-section {
    padding: 12px 16px 5px;
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--tx3); flex-shrink: 0;
  }

  .sb-list { flex:1; overflow-y:auto; padding:3px 8px; }

  .sb-item {
    padding: 8px 10px;
    border-radius: 9px;
    cursor: pointer;
    display: flex; align-items: center; gap: 9px;
    color: var(--tx2);
    font-size: 13px;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap; overflow: hidden;
  }
  .sb-item:hover { background: var(--hover); color: var(--tx); }
  .sb-item.active { background: rgba(99,102,241,0.12); color: #818cf8; }
  .sb-item svg { flex-shrink:0; opacity:0.6; width:14px; height:14px; }
  .sb-item span { overflow:hidden; text-overflow:ellipsis; }

  .sb-footer {
    border-top: 1px solid var(--bd);
    padding: 12px;
    display: flex; align-items: center; gap: 9px;
    flex-shrink: 0;
  }
  .u-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    font-size:11px;font-weight:700;color:#fff;flex-shrink:0;
  }
  .u-name { font-size:13px;font-weight:600;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
  .u-role { font-size:11px;color:var(--tx3);text-transform:capitalize; }
  .sb-icon-btn {
    background:none;border:none;cursor:pointer;
    color:var(--tx3);padding:6px;border-radius:8px;
    display:flex;align-items:center;justify-content:center;
    transition:color 0.15s,background 0.15s;
    flex-shrink:0;
  }
  .sb-icon-btn:hover { color:var(--tx);background:var(--hover); }

  /* ── main ── */
  .main-col { flex:1; display:flex;flex-direction:column;overflow:hidden;min-width:0; }

  .topbar {
    height: 56px; flex-shrink:0;
    display:flex;align-items:center;gap:10px;padding:0 18px;
    background:var(--surface);
    border-bottom:1px solid var(--bd);
  }
  .tb-title { font-size:15px;font-weight:700;color:var(--tx);flex:1;letter-spacing:-0.2px; }
  .tb-title span { color:#818cf8; }
  .tb-actions { display:flex;align-items:center;gap:8px; }
  .icon-btn {
    width:32px;height:32px;border-radius:8px;
    background:none;border:1px solid var(--bd);
    color:var(--tx2);cursor:pointer;
    display:flex;align-items:center;justify-content:center;
    transition:background 0.15s,color 0.15s,border-color 0.15s;
  }
  .icon-btn:hover { background:var(--hover);color:var(--tx);border-color:var(--bd2); }
  .icon-btn svg { width:14px;height:14px; }
  .user-badge {
    display:flex;align-items:center;gap:7px;
    padding:4px 10px 4px 5px;
    background:var(--surface2);
    border:1px solid var(--bd);border-radius:20px;
    cursor:default;
  }
  .ub-av {
    width:22px;height:22px;border-radius:50%;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    font-size:9px;font-weight:700;color:#fff;flex-shrink:0;
  }
  .ub-name { font-size:12px;font-weight:600;color:var(--tx2); }

  /* ── page content ── */
  .page-outlet { flex:1;overflow:hidden;display:flex;flex-direction:column; }

  /* ── themes ── */
  :root[data-theme="dark"] {
    --sidebar-bg:#0a0a0e;
    --surface:#0f0f14;
    --surface2:#141418;
    --hover:rgba(255,255,255,0.05);
    --bd:rgba(255,255,255,0.07);
    --bd2:rgba(255,255,255,0.13);
    --tx:#f0f0f5;
    --tx2:#8b8fa8;
    --tx3:#44475a;
  }
  :root[data-theme="light"] {
    --sidebar-bg:#fafafa;
    --surface:#ffffff;
    --surface2:#f4f4f7;
    --hover:rgba(0,0,0,0.04);
    --bd:rgba(0,0,0,0.08);
    --bd2:rgba(0,0,0,0.14);
    --tx:#111118;
    --tx2:#555570;
    --tx3:#aaaabc;
  }
`;

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { dark, toggle } = useTheme();
  const [open, setOpen] = useState(true);
  const [activeChat, setActiveChat] = useState(null);
  const [userChats, setUserChats] = useState([]);

  const username = localStorage.getItem("username") || "";
  const role     = localStorage.getItem("role") || "member";
  const initials = username ? username.slice(0, 2).toUpperCase() : "?";

  const refreshChats = useCallback(() => {
    setUserChats(loadUserChats(username, role));
  }, [username, role]);

  useEffect(() => {
    refreshChats();
    window.addEventListener("nexusai_chats_updated", refreshChats);
    return () => window.removeEventListener("nexusai_chats_updated", refreshChats);
  }, [refreshChats]);

  const openLatestOrNewChat = () => {
    if (userChats.length > 0) {
      const latest = userChats[0];
      setActiveChat(latest.id);
      navigate("/ask", { state: { chatId: latest.id } });
      return;
    }
    setActiveChat(null);
    navigate("/ask", { state: { mode: "new" } });
  };

  const logout = () => { localStorage.clear(); navigate("/"); };
  const isAsk  = location.pathname === "/ask";

  return (
    <>
      <style>{CSS}</style>
      <div className="layout-root">

        {/* ── Sidebar ────────────────── */}
        <aside className={`sidebar ${open ? "" : "closed"}`}>
          {/* Brand */}
          <div className="sb-head">
            <div className="sb-logo">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <span className="sb-brand">Nexus<span>AI</span></span>
          </div>

          {/* New chat */}
          <button className="sb-new-btn" onClick={() => { setActiveChat(null); navigate("/ask", { state: { mode: "new" } }); }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            New conversation
          </button>

          {/* Nav */}
          <div className="sb-section">Navigation</div>
          <div style={{ padding: "3px 8px", flexShrink: 0 }}>
            <div className={`sb-item ${location.pathname==="/dashboard"?"active":""}`} onClick={() => navigate("/dashboard")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
              <span>Dashboard</span>
            </div>
            <div className={`sb-item ${location.pathname==="/ask"&&!activeChat?"active":""}`} onClick={openLatestOrNewChat}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <span>AI Chat Assistant</span>
            </div>
          </div>

          {/* Recent chats */}
          {userChats.length > 0 && <div className="sb-section">Recent Chats</div>}
          <div className="sb-list">
            {userChats.length > 0 ? (
              userChats.map(c => (
                <div
                  key={c.id}
                  className={`sb-item ${activeChat === c.id ? "active" : ""}`}
                  onClick={() => { setActiveChat(c.id); navigate("/ask", { state: { chatId: c.id } }); }}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                  </svg>
                  <span>{c.title}</span>
                </div>
              ))
            ) : null}
          </div>

          {/* Footer */}
          <div className="sb-footer">
            <div className="u-avatar">{initials}</div>
            <div style={{ flex: 1, overflow: "hidden" }}>
              <div className="u-name">{username}</div>
              <div className="u-role">{role}</div>
            </div>
            <button className="sb-icon-btn" onClick={toggle} title="Toggle theme">
              {dark
                ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
                : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
              }
            </button>
            <button className="sb-icon-btn" onClick={logout} title="Logout">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>
              </svg>
            </button>
          </div>
        </aside>

        {/* ── Main Column ─────────────── */}
        <div className="main-col">
          {/* Topbar */}
          <header className="topbar">
            <button className="icon-btn" onClick={() => setOpen(o => !o)}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
              </svg>
            </button>

            <span className="tb-title">
              {isAsk ? <>AI Chat <span>Assistant</span></> : "Dashboard"}
            </span>

            <div className="tb-actions">
              <button className="icon-btn" onClick={toggle}>
                {dark
                  ? <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/></svg>
                  : <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
                }
              </button>
              <div className="user-badge">
                <div className="ub-av">{initials}</div>
                <span className="ub-name">{username}</span>
              </div>
            </div>
          </header>

          <div className="page-outlet">
            <Outlet />
          </div>
        </div>
      </div>
    </>
  );
}