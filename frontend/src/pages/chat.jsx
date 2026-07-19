import { useEffect, useState } from "react";
import { useAuth } from "../contexts/AuthContext";

import useVerifySession from "../hooks/use_verify_session";
import { useNavigate } from "react-router-dom";

import { askQuestion } from "../services/auth";


    const MAX_CHARACTERS = 2000;

    // Maximum number of complete question-answer exchanges
    const MAX_EXCHANGES = 5;
    

function Chat() {

    const { user } = useAuth();

    const navigate = useNavigate();
    const verifySession = useVerifySession();

    const [checking, setChecking] = useState(true);

    const [messages, setMessages] = useState([]);
    const [historyLoaded, setHistoryLoaded] = useState(false);
    const [input, setInput] = useState("");

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const [showDocuments, setShowDocuments] = useState(false);

    // ==============================
    // Chat Configuration
    // ==============================

    // Separate history for every logged-in user
 const CHAT_STORAGE_KEY = user
    ? `chat_history_${user.emp_id}`
    : null;

    // ==============================
    // Restore chat history
    // Runs once when the page loads
    // ==============================

useEffect(() => {

    if (!CHAT_STORAGE_KEY) {

        return;

    }

    const savedMessages = sessionStorage.getItem(CHAT_STORAGE_KEY);

    if (savedMessages) {

        setMessages(JSON.parse(savedMessages));

    }

    setHistoryLoaded(true);

}, [CHAT_STORAGE_KEY]);

    // ==============================
    // Save history whenever it changes
    // ==============================

useEffect(() => {
    // Only write to sessionStorage if the user is logged in
    if (!CHAT_STORAGE_KEY || !historyLoaded || !user) {
        return;
    }

    sessionStorage.setItem(
        CHAT_STORAGE_KEY,
        JSON.stringify(messages)
    );
}, [messages, CHAT_STORAGE_KEY, historyLoaded, user]);


    // ==============================
    // Verify user session
    // ==============================

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

    if (checking) {

        return <h2>Loading...</h2>;

    }

    // ==============================
    // Send Message
    // ==============================

    const handleSend = async () => {

        if (!input.trim()) {

            return;

        }

        const question = input.trim();

        const userMessage = {

            role: "user",

            content: question,

        };

        // Show user's message immediately

        setMessages((previousMessages) => [

            ...previousMessages,

            userMessage,

        ]);

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

            setMessages((previousMessages) => {

                const updatedMessages = [

                    ...previousMessages,

                    assistantMessage,

                ];

                // One exchange = User + Assistant = 2 messages
                // Keep only the most recent exchanges

                if (updatedMessages.length > MAX_EXCHANGES * 2) {

                    updatedMessages.splice(0, 2);

                }

                return updatedMessages;

            });

        }

        catch (err) {

            setError(

                err.response?.data?.detail ||

                "Failed to get a response."

            );

        }

        finally {

            setLoading(false);

        }

    };

    return (

        <main>

            <section>

                {/* Header */}

                <h2>

                    Role-Based AI Assistant

                </h2>

                <button>

                    Documents

                </button>

            </section>

            {messages.length === 0 && (

                <section>

                    <h2>

                        Ready when you are, {user.name}.

                    </h2>

                    <p>

                        Ask a question about the documents you have access to.

                    </p>

                </section>

            )}

            <section>

                {messages.map((message, index) => (

                    <div key={index}>

                        <strong>

                            {message.role === "user" ? "You" : "Assistant"}

                        </strong>

                        <p>

                            {message.content}

                        </p>

                    </div>

                ))}

            </section>

            <section>

                {

                    error && (

                        <p>

                            {error}

                        </p>

                    )

                }

                <textarea

                    value={input}

                    maxLength={MAX_CHARACTERS}

                    onChange={(event) => setInput(event.target.value)}

                    placeholder="Ask a question about your company documents..."

                >

                </textarea>

                <p

                    style={{

                        color:

                            input.length >= MAX_CHARACTERS

                                ? "red"

                                : "inherit",

                    }}

                >

                    {input.length}/{MAX_CHARACTERS}

                </p>

                <button

                    onClick={handleSend}

                    disabled={!input.trim() || loading}

                >

                    {loading ? "Thinking..." : "Send"}

                </button>

            </section>

        </main>

    );

}

export default Chat;