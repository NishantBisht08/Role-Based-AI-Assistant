/**
 * CopyButton — One-click clipboard copy with visual feedback
 *
 * Used in the Demo Guide for copying Employee IDs and passwords.
 */
import { useState } from "react";
import Button from "../ui/Button";

function CopyButton({ text, label = "Copy" }) {
    const [copied, setCopied] = useState(false);

    async function handleCopy() {
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // Fallback for environments without clipboard API
            const textarea = document.createElement("textarea");
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            document.body.removeChild(textarea);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    }

    return (
        <Button variant="ghost" size="sm" onClick={handleCopy}>
            {copied ? "Copied!" : label}
        </Button>
    );
}

export default CopyButton;
