/**
 * FlipCard — 3D flip interaction for "Did You Know?" facts
 *
 * Front shows a blank premium card with a hint; back reveals the fact.
 * Click toggles the flip animation using CSS 3D transforms.
 */
import { useState } from "react";

function FlipCard({ frontTitle = "Did You Know?", backTitle, backDescription }) {
    const [flipped, setFlipped] = useState(false);

    return (
        <div
            className={`flip-card ${flipped ? "flipped" : ""}`}
            onClick={() => setFlipped((prev) => !prev)}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                    setFlipped((prev) => !prev);
                }
            }}
        >
            <div className="flip-card__inner">
                <div className="flip-card__front">
                    <span style={{ fontSize: "2rem" }}>💡</span>
                    <h4>{frontTitle}</h4>
                    <p className="flip-card__hint">Click to reveal</p>
                </div>
                <div className="flip-card__back">
                    <h4>{backTitle}</h4>
                    <p>{backDescription}</p>
                </div>
            </div>
        </div>
    );
}

export default FlipCard;
