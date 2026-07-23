/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useState } from "react";

import { getCurrentUser } from "../services/auth";


const AuthContext = createContext();

export function AuthProvider({ children }) {

    const [user, setUser] = useState(null);

    const [loading, setLoading] = useState(true);

    useEffect(() => {

    async function fetchUser() {

        try {

            const currentUser = await getCurrentUser();

            setUser(currentUser);

        }

        catch {

            setUser(null);

        }

        finally {

            setLoading(false);

        }

    }

    fetchUser();

}, []);

return (
    <AuthContext.Provider
        value={{
            user,
            setUser,
            loading,
        }}
    >
        {children}
    </AuthContext.Provider>
);

}

export function useAuth() {
    return useContext(AuthContext);
}
