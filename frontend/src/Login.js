import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const S = {
  page: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#0d0d0f",
    position: "relative",
    overflow: "hidden",
  },
  orb1: {
    position: "absolute", borderRadius: "50%", pointerEvents: "none",
    width: 600, height: 600, top: -200, right: -150,
    background: "radial-gradient(circle, rgba(99,102,241,0.22) 0%, transparent 65%)",
    animation: "orbPulse 9s ease-in-out infinite alternate",
  },
  orb2: {
    position: "absolute", borderRadius: "50%", pointerEvents: "none",
    width: 500, height: 500, bottom: -160, left: -120,
    background: "radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 65%)",
    animation: "orbPulse 12s ease-in-out infinite alternate-reverse",
  },
  orb3: {
    position: "absolute", borderRadius: "50%", pointerEvents: "none",
    width: 300, height: 300, top: "40%", left: "30%",
    background: "radial-gradient(circle, rgba(79,70,229,0.1) 0%, transparent 70%)",
    animation: "orbPulse 7s ease-in-out infinite alternate",
  },
  grid: {
    position: "absolute", inset: 0, pointerEvents: "none",
    backgroundImage: `linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)`,
    backgroundSize: "50px 50px",
  },
  card: {
    position: "relative", zIndex: 10,
    width: "100%", maxWidth: 420,
    margin: "0 16px",
    background: "rgba(18,18,24,0.85)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 24,
    padding: "40px 36px",
    backdropFilter: "blur(20px)",
    WebkitBackdropFilter: "blur(20px)",
    boxShadow: "0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(99,102,241,0.08)",
    animation: "cardIn 0.55s cubic-bezier(0.4,0,0.2,1)",
  },
  logoRow: {
    display: "flex", alignItems: "center", gap: 10, marginBottom: 32,
  },
  logoBox: {
    width: 40, height: 40, borderRadius: 12,
    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    display: "flex", alignItems: "center", justifyContent: "center",
    boxShadow: "0 0 20px rgba(99,102,241,0.4)",
  },
  brandName: {
    fontSize: 18, fontWeight: 700, color: "#f0f0f5", letterSpacing: "-0.3px",
  },
  brandAccent: { color: "#818cf8" },
  heading: {
    fontSize: 28, fontWeight: 700, color: "#f0f0f5",
    letterSpacing: "-0.6px", marginBottom: 6,
  },
  subheading: {
    fontSize: 14, color: "#6b7280", marginBottom: 32, lineHeight: 1.6,
  },
  label: {
    display: "block", fontSize: 11, fontWeight: 600,
    textTransform: "uppercase", letterSpacing: "0.07em",
    color: "#6b7280", marginBottom: 7,
  },
  inputWrap: { marginBottom: 16, position: "relative" },
  input: {
    width: "100%", padding: "12px 14px",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 12, outline: "none",
    fontSize: 14, color: "#f0f0f5",
    fontFamily: "inherit",
    transition: "border-color 0.2s, box-shadow 0.2s",
  },
  btn: {
    width: "100%", padding: "13px",
    background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    border: "none", borderRadius: 12,
    fontSize: 14, fontWeight: 600, color: "#fff",
    cursor: "pointer", marginTop: 8,
    transition: "opacity 0.2s, transform 0.15s, box-shadow 0.2s",
    letterSpacing: "0.2px",
    boxShadow: "0 0 0 0 rgba(99,102,241,0)",
  },
  errorBox: {
    marginTop: 14, padding: "10px 13px",
    background: "rgba(239,68,68,0.1)",
    border: "1px solid rgba(239,68,68,0.25)",
    borderRadius: 10, fontSize: 13, color: "#f87171",
    display: "flex", alignItems: "center", gap: 8,
  },
};

const CSS = `
  @keyframes orbPulse {
    from { transform: scale(1) translate(0,0); opacity:0.8; }
    to   { transform: scale(1.12) translate(15px,10px); opacity:1; }
  }
  @keyframes cardIn {
    from { opacity:0; transform: translateY(28px) scale(0.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
  }
  .login-input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
  }
  .login-btn:hover:not(:disabled) {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
  }
  .login-btn:active { transform: translateY(0) !important; }
  .login-btn:disabled { opacity: 0.55; cursor: not-allowed; }
`;

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!username.trim() || !password.trim()) {
      setError("Please fill in both fields."); return;
    }
    setError(""); setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/login", { username, password });
      localStorage.setItem("access_token", res.data.access_token);
      localStorage.setItem("refresh_token", res.data.refresh_token);
      localStorage.setItem("role", res.data.role);
      localStorage.setItem("username", username);
      navigate("/dashboard");
    } catch {
      setError("Invalid username or password. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{CSS}</style>
      <div style={S.page}>
        <div style={S.orb1} /><div style={S.orb2} /><div style={S.orb3} />
        <div style={S.grid} />

        <div style={S.card}>
          {/* Brand */}
          <div style={S.logoRow}>
            <div style={S.logoBox}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <span style={S.brandName}>Nexus<span style={S.brandAccent}>AI</span></span>
          </div>

          <h1 style={S.heading}>Welcome back</h1>
          <p style={S.subheading}>Sign in to your AI knowledge assistant</p>

          {/* Username */}
          <div style={S.inputWrap}>
            <label style={S.label}>Username</label>
            <input
              className="login-input"
              style={S.input}
              placeholder="Enter username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleLogin()}
              autoComplete="username"
            />
          </div>

          {/* Password */}
          <div style={S.inputWrap}>
            <label style={S.label}>Password</label>
            <input
              className="login-input"
              style={S.input}
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleLogin()}
              autoComplete="current-password"
            />
          </div>

          {error && (
            <div style={S.errorBox}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              {error}
            </div>
          )}

          <button
            className="login-btn"
            style={S.btn}
            onClick={handleLogin}
            disabled={loading}
          >
            {loading ? "Signing in…" : "Sign in →"}
          </button>
        </div>
      </div>
    </>
  );
}