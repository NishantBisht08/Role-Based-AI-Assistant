/**
 * ThemeToggle — Global light/dark mode switcher
 *
 * Placed in the Navbar. Reads/writes theme via ThemeContext and
 * persists the choice in localStorage.
 */
import { useTheme } from "../contexts/ThemeContext";

function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
        >
            {theme === "light" ? "🌙" : "☀️"}
        </button>
    );
}

export default ThemeToggle;
