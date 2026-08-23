import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { getCurrentUser } from "../services/auth";

const AuthContext = createContext();

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // 1. We create the reusable verify function here
    const verifySession = useCallback(async () => {
        try {
            const currentUser = await getCurrentUser();
            setUser(currentUser);
            return true;
        } catch {
            setUser(null);
            return false;
        }
    }, []);

    // 2. We use it for the initial load! (Code reuse!)
    useEffect(() => {
        async function init() {
            await verifySession();
            setLoading(false);
        }
        init();
    }, [verifySession]);

    return (
        <AuthContext.Provider
            value={{
                user,
                setUser,
                loading,
                verifySession, // 3. We expose it so ProtectedRoute can use it
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    return useContext(AuthContext);
}