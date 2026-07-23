import { useCallback } from "react";

import { useAuth } from "../contexts/AuthContext";
import { getCurrentUser } from "../services/auth";

function useVerifySession() {

    const { setUser } = useAuth();

    const verifySession = useCallback(async () => {

        try {

            const currentUser = await getCurrentUser();

            setUser(currentUser);

            return true;

        }

        catch {

            setUser(null);

            return false;

        }

    }, [setUser]);

    return verifySession;

}

export default useVerifySession;