/**
 * MessageBubble — Individual chat message display
 *
 * Renders user and assistant messages with distinct styling,
 * matching modern AI chat interfaces (ChatGPT / Claude aesthetic).
 */
function MessageBubble({ role, content }) {
    const isUser = role === "user";
    const initials = isUser ? "You" : "AI";

    return (
        <div className={`chat-message ${isUser ? "chat-message--user" : "chat-message--assistant"}`}>
            <div className="chat-message__avatar">{initials === "You" ? "Y" : "N"}</div>
            <div className="chat-message__bubble">{content}</div>
        </div>
    );
}

export default MessageBubble;
