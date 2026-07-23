/**
 * ParchmentPaper — Document tile styled as aged parchment
 *
 * Used on Dataset pages. Soft yellow paper with subtle hover movement.
 * Click triggers the parent document viewer via onClick callback.
 */
function ParchmentPaper({ title, onClick }) {
    return (
        <div className="parchment-paper" onClick={onClick} role="button" tabIndex={0}>
            <span className="parchment-paper__title">{title}</span>
        </div>
    );
}

export default ParchmentPaper;
