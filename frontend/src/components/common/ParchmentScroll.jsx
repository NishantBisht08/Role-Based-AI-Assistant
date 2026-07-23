/**
 * ParchmentScroll — Cinematic modal for Example Questions
 *
 * Renders example questions inside an aged parchment scroll overlay
 * with blurred backdrop. Closes via the top-right button or backdrop click.
 */
import { exampleQuestions } from "../../data/exampleQuestions";

function ParchmentScroll({ onClose }) {
    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div
                className="modal-content"
                onClick={(event) => event.stopPropagation()}
            >
                <button className="modal-close" onClick={onClose} aria-label="Close">
                    ✕
                </button>
                <div className="parchment-scroll">
                    <h2>Example Questions</h2>
                    <p style={{ textAlign: "center", marginBottom: "1.5rem", fontStyle: "italic" }}>
                        Select a question from the scroll to inspire your queries.
                    </p>

                    {Object.values(exampleQuestions).map((category) => (
                        <section key={category.title}>
                            <h3>{category.title}</h3>
                            {category.sections.map((section) => (
                                <section key={section.title}>
                                    <h3 style={{ fontSize: "1rem" }}>{section.title}</h3>
                                    <ol>
                                        {section.questions.map((question, index) => (
                                            <li key={index}>{question}</li>
                                        ))}
                                    </ol>
                                </section>
                            ))}
                        </section>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default ParchmentScroll;
