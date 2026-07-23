/**
 * AuthCard — Shared layout for authentication-related forms
 *
 * Provides the centered glassmorphism card used on Login, Set Password,
 * Change Password, and Create User pages for visual consistency.
 */
function AuthCard({ title, subtitle, children, footer }) {
    return (
        <div className="auth-page">
            <div className="auth-card">
                <div className="auth-card__header">
                    <h2>{title}</h2>
                    {subtitle && <p>{subtitle}</p>}
                </div>
                {children}
                {footer && <div className="auth-card__footer">{footer}</div>}
            </div>
        </div>
    );
}

export default AuthCard;
