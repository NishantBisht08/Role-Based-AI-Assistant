/**
 * Chat — AI conversation interface (NovaMind)
 *
 * Modern chat UI inspired by ChatGPT/Claude while preserving
 * all existing backend integration: askQuestion(), sessionStorage
 * history, character limits, and conversation trimming.
 */
import { useEffect, useState, useRef } from "react";
import { useAuth } from "../contexts/AuthContext";
import useVerifySession from "../hooks/use_verify_session";
import { useNavigate } from "react-router-dom";
import { askQuestion } from "../services/auth";
import ParchmentScroll from "../components/common/ParchmentScroll";
import MessageBubble from "../components/chat/MessageBubble";
import Button from "../components/ui/Button";
import LoadingSpinner from "../components/ui/LoadingSpinner";

const MAX_CHARACTERS = 2000;
const MAX_EXCHANGES = 5;

function Chat() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const verifySession = useVerifySession();
    const messagesEndRef = useRef(null);

    const [checking, setChecking] = useState(true);
    const [messages, setMessages] = useState([]);
    const [historyLoaded, setHistoryLoaded] = useState(false);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [showQuestions, setShowQuestions] = useState(false);
    const [hasTyped, setHasTyped] = useState(false);

    const CHAT_STORAGE_KEY = user ? `chat_history_${user.emp_id}` : null;

    // Restore chat history from sessionStorage.
    // This intentionally hydrates UI state from browser session storage; malformed
    // data is discarded so a bad local value cannot break the chat page.
    useEffect(() => {
        if (!CHAT_STORAGE_KEY) return;
        const savedMessages = sessionStorage.getItem(CHAT_STORAGE_KEY);
        if (savedMessages) {
            try {
                // eslint-disable-next-line react-hooks/set-state-in-effect
                setMessages(JSON.parse(savedMessages));
            } catch {
                sessionStorage.removeItem(CHAT_STORAGE_KEY);
            }
        }
        setHistoryLoaded(true);
    }, [CHAT_STORAGE_KEY]);

    // Persist chat history on every change
    useEffect(() => {
        if (!CHAT_STORAGE_KEY || !historyLoaded || !user) return;
        sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
    }, [messages, CHAT_STORAGE_KEY, historyLoaded, user]);

    // Verify session on mount
    useEffect(() => {
        async function checkSession() {
            const valid = await verifySession();
            if (!valid) {
                navigate("/login", { replace: true });
                return;
            }
            setChecking(false);
        }
        checkSession();
    }, [navigate, verifySession]);

    // Auto-scroll to latest message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, loading]);

    if (checking) {
        return <LoadingSpinner text="Loading chat..." />;
    }

    const handleSend = async () => {
        if (!input.trim()) return;

        const question = input.trim();
        const userMessage = { role: "user", content: question };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setLoading(true);
        setError("");

        try {
            const response = await askQuestion(question);
            const assistantMessage = {
                role: "assistant",
                content: response.answer,
                sources: response.sources,
            };

            setMessages((prev) => {
                const updated = [...prev, assistantMessage];
                if (updated.length > MAX_EXCHANGES * 2) {
                    updated.splice(0, 2);
                }
                return updated;
            });
        } catch (err) {
            setError(err.response?.data?.detail || "Failed to get a response.");
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="chat-page">
            {/* Header with project title and example questions */}
            <div className="chat-header">
                <h2>Novaris</h2>
                <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setShowQuestions(true)}
                >
                    📜 Example Questions
                </Button>
            </div>

            {/* Parchment scroll modal for example questions */}
            {showQuestions && (
                <ParchmentScroll onClose={() => setShowQuestions(false)} />
            )}

            {/* Welcome message — disappears after first message */}
            {messages.length === 0 && (
                <div className="chat-welcome">
                    <h2>Ready when you are, {user.name}.</h2>
                    {!hasTyped && (
                        <p>Ask a question about the documents you have access to.</p>
                    )}
                </div>
            )}

            {/* Message thread */}
            {messages.length > 0 && (
                <div className="chat-messages">
                    {messages.map((message, index) => (
                        <MessageBubble
                            key={index}
                            role={message.role}
                            content={message.content}
                        />
                    ))}
                    {loading && (
                        <div className="chat-message chat-message--assistant">
                            <div className="chat-message__avatar">N</div>
                            <div className="chat-message__bubble">
                                <span style={{ opacity: 0.6 }}>Thinking...</span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            )}

            {/* Input area — centered with character counter */}
            <div className="chat-input-area">
                {error && <p className="form-error" style={{ marginBottom: "0.5rem" }}>{error}</p>}

                <div className="chat-input-wrapper">
                    <textarea
                        value={input}
                        maxLength={MAX_CHARACTERS}
                        onChange={(event) => {
                            setInput(event.target.value);
                            if (!hasTyped) setHasTyped(true);
                        }}
                        onKeyDown={handleKeyDown}
                        placeholder={
                            hasTyped
                                ? "Ask a question about your company documents..."
                                : "Ask a question about the documents you have access to."
                        }
                        rows={1}
                    />
                    <Button
                        variant="primary"
                        onClick={handleSend}
                        disabled={!input.trim() || loading}
                    >
                        {loading ? "..." : "Send"}
                    </Button>
                </div>

                <div className="chat-input-meta">
                    <span
                        className={`chat-char-count ${
                            input.length >= MAX_CHARACTERS ? "chat-char-count--limit" : ""
                        }`}
                    >
                        {input.length}/{MAX_CHARACTERS}
                    </span>
                </div>
            </div>
        </div>
    );
}

export default Chat;

