/**
 * ProtectedRoute — Route guard for authenticated pages
 *
 * Redirects unauthenticated users to /login.
 * Shows a loading spinner while AuthContext initializes.
 */

import { useEffect, useState, useRef } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import LoadingSpinner from "./ui/LoadingSpinner";

function ProtectedRoute({ children }) {
    const { user, loading: authLoading, verifySession } = useAuth();
    const [verifying, setVerifying] = useState(true);
    const location = useLocation();

    // 1. Ref to track user without causing dependency loops
    const userRef = useRef(user);

    // 2. Keep ref updated safely in an effect (after render)
    useEffect(() => {
        userRef.current = user;
    }, [user]);

    // 3. Verification effect with cancellation cleanup for race conditions
    useEffect(() => {
        if (authLoading) return;

        let cancelled = false;

        if (!userRef.current) {
            setVerifying(false);
            return;
        }

        async function runVerification() {
            await verifySession();

            if (!cancelled) {
                setVerifying(false);
            }
        }

        runVerification();

        return () => {
            cancelled = true;
        };
    }, [location.pathname, verifySession, authLoading]);

    if (authLoading || verifying) {
        return <LoadingSpinner text="Verifying session..." />;
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return children;
}

export default ProtectedRoute;

