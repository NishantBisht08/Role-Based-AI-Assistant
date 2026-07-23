import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { AuthProvider } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";

// Application entry point — wraps the app with global providers.
// AuthProvider manages authentication state; ThemeProvider handles light/dark mode.
createRoot(document.getElementById("root")).render(
    <StrictMode>
        <ThemeProvider>
            <AuthProvider>
                <App />
            </AuthProvider>
        </ThemeProvider>
    </StrictMode>
);
