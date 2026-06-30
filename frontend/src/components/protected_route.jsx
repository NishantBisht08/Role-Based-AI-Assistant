import { Navigate } from "react-router-dom";

import { useAuth } from "../contexts/AuthContext";


function ProtectedRoute({ children }) {

    const { user, loading } = useAuth();       //authentication state

    if (loading) 
        {

          return <h2>Loading...</h2>;

        }

    if (!user) {

            return <Navigate to="/login" replace />;

         }

    return children;

}

export default ProtectedRoute;