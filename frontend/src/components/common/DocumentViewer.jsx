/**
 * DocumentViewer — Full-screen modal for reading document contents
 *
 * Opens over a blurred backdrop when a parchment paper is clicked.
 * Displays the document title and raw text content with a close button.
 */
function DocumentViewer({ title, content, onClose }) {
    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div
                className="modal-content"
                onClick={(event) => event.stopPropagation()}
            >
                <button className="modal-close" onClick={onClose} aria-label="Close">
                    ✕
                </button>
                <div className="document-viewer">
                    <h2>{title}</h2>
                    <div className="document-viewer__content">{content}</div>
                </div>
            </div>
        </div>
    );
}

export default DocumentViewer;
