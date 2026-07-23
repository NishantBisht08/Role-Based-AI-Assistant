/**
 * GateEntrance — Medieval vault gate for Demo Guide navigation
 *
 * Inspired by adventure game entrances (Uncharted / TLOU aesthetic).
 * On click, the gate doors animate open before navigating to /demo.
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";

function GateEntrance() {
    const navigate = useNavigate();
    const [opening, setOpening] = useState(false);

    function handleClick() {
        if (opening) return;
        setOpening(true);
        // Navigate after gate animation completes (~600ms)
        setTimeout(() => navigate("/demo"), 650);
    }

    return (
        <button
            className={`gate-entrance ${opening ? "opening" : ""}`}
            onClick={handleClick}
            aria-label="Enter Demo Guide"
        >
            <div className="gate-entrance__frame">
                <div className="gate-entrance__arch" />
                <div className="gate-entrance__door gate-entrance__door--left" />
                <div className="gate-entrance__door gate-entrance__door--right" />
            </div>
            <span className="gate-entrance__label">Enter Demo Guide</span>
        </button>
    );
}

export default GateEntrance;
