import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import LoadingSpinner from "./ui/LoadingSpinner";

/**
 * ProtectedRoute — Route guard for authenticated pages
 *
 * Redirects unauthenticated users to /login.
 * Shows a loading spinner while AuthContext initializes.
 */
function ProtectedRoute({ children }) {
    const { user, loading } = useAuth();

    if (loading) {
        return <LoadingSpinner text="Verifying session..." />;
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return children;
}

export default ProtectedRoute;
