/* eslint-disable react-refresh/only-export-components */
/**
 * ThemeContext — Global Light/Dark Theme Management
 *
 * Persists the user's theme preference in localStorage so it survives
 * page refreshes. Applies `data-theme` on <html> for CSS variable switching.
 *
 * Note: Only UI preference is stored here — never authentication tokens.
 */
import { createContext, useContext, useEffect, useState } from "react";

const ThemeContext = createContext();

const STORAGE_KEY = "novaris-theme";

export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState(() => {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved === "light" || saved === "dark") {
            return saved;
        }
        // Respect system preference on first visit
        return window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";
    });

    useEffect(() => {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem(STORAGE_KEY, theme);
    }, [theme]);

    function toggleTheme() {
        setTheme((prev) => (prev === "light" ? "dark" : "light"));
    }

    return (
        <ThemeContext.Provider value={{ theme, setTheme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    return useContext(ThemeContext);
}

