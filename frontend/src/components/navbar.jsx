/**
 * Navbar — Global navigation bar for Novaris
 *
 * Sticky header with brand, route links, theme toggle, and auth actions.
 * Dashboard link only appears when a user is authenticated.
 * Logout clears sessionStorage chat history (preserving existing behavior).
 */
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { logout } from "../services/auth";
import ThemeToggle from "./theme_toggle";
import Button from "./ui/Button";

function Navbar() {
    const navigate = useNavigate();
    const { user, setUser } = useAuth();

    const handleLogout = async () => {
        const confirmed = window.confirm(
            "Are you sure you want to log out?"
        );

        if (!confirmed) {
            return;
        }

        try {
            await logout();
        } catch (err) {
            console.error("Backend logout failed:", err);
        } finally {
            if (user?.emp_id) {
                sessionStorage.removeItem(`chat_history_${user.emp_id}`);
            }
            setUser(null);
            navigate("/");
        }
    };

    return (
        <header className="navbar">
            <div className="navbar__inner">
                {/* Brand */}
                <NavLink to="/" className="navbar__brand">
                    <span className="navbar__brand-icon">N</span>
                    Novaris
                </NavLink>

                {/* Navigation links */}
                <nav className="navbar__links">
                    <NavLink
                        to="/"
                        className={({ isActive }) =>
                            `navbar__link ${isActive ? "active" : ""}`
                        }
                        end
                    >
                        Home
                    </NavLink>

                    {user && (
                        <NavLink
                            to="/dashboard"
                            className={({ isActive }) =>
                                `navbar__link ${isActive ? "active" : ""}`
                            }
                        >
                            Dashboard
                        </NavLink>
                    )}

                    <NavLink
                        to="/about"
                        className={({ isActive }) =>
                            `navbar__link ${isActive ? "active" : ""}`
                        }
                    >
                        About
                    </NavLink>
                </nav>

                {/* Right-side actions: theme toggle + auth */}
                <div className="navbar__actions">
                    <ThemeToggle />

                    {user ? (
                        <>
                            <span className="navbar__user">{user.name}</span>
                            <Button variant="secondary" size="sm" onClick={handleLogout}>
                                Logout
                            </Button>
                        </>
                    ) : (
                        <Button
                            variant="primary"
                            size="sm"
                            onClick={() => navigate("/login")}
                        >
                            Login
                        </Button>
                    )}
                </div>
            </div>
        </header>
    );
}

export default Navbar;
