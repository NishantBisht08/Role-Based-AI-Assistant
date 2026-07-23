/**
 * LoadingSpinner — Consistent loading state across protected pages
 *
 * Displayed while AuthContext or session verification is in progress.
 */
function LoadingSpinner({ text = "Loading..." }) {
    return (
        <div className="loading-spinner">
            <div className="loading-spinner__ring" />
            <p className="loading-spinner__text">{text}</p>
        </div>
    );
}

export default LoadingSpinner;
