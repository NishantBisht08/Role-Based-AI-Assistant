import { useState, useEffect } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import useVerifySession from "../hooks/use_verify_session";
import { useAuth } from "../contexts/AuthContext";
import { createUser } from "../services/auth";

function CreateUser() {

    const navigate = useNavigate();

    // Currently authenticated user
    const { user, loading } = useAuth();

    // Hook used to verify whether the current session is still valid
    const verifySession = useVerifySession();

    // Prevent page rendering until session verification finishes
    const [checking, setChecking] = useState(true);

    const [empId, setEmpId] = useState("");
    const [name, setName] = useState("");
    const [role, setRole] = useState("employee");

    const [message, setMessage] = useState("");
    const [error, setError] = useState("");

    // Tracks whether the create user request is in progress
    const [submitting, setSubmitting] = useState(false);

    // Verify the session every time this protected page is opened.
    // If the refresh token has expired, redirect the user to login
    // before showing the page.
    useEffect(() => {

        async function checkSession() {

            const valid = await verifySession();

            if (!valid) {

                navigate("/login", { replace: true });

                return;

            }

            setChecking(false);

        }

        checkSession();

    }, [navigate, verifySession]);

    // Wait until both AuthContext and session verification finish
    // before rendering the page.
    if (loading || checking) {

        return <h2>Loading...</h2>;

    }

    // Only administrators are allowed to access this page.
    if (user?.role !== "admin") {

        return <Navigate to="/dashboard" replace />;

    }

    async function handleSubmit(e) {

        e.preventDefault();

        setMessage("");
        setError("");

        // Prevent duplicate requests while the current
        // create user operation is still in progress.
        setSubmitting(true);

        try {

            const response = await createUser(
                empId,
                name,
                role
            );

            setMessage(response.message);

            // Clear the form after successful user creation
            setEmpId("");
            setName("");
            setRole("employee");

        }

        catch (err) {

            setError(
                err.response?.data?.detail ||
                "Failed to create user."
            );

        }

        finally {

            // Re-enable the form regardless of success or failure
            setSubmitting(false);

        }

    }

    return (

        <main>

            <h2>Create User</h2>

            <form onSubmit={handleSubmit}>

                <div>

                    <label>
                        Employee ID
                    </label>

                    <input
                        type="text"
                        value={empId}
                        onChange={(e) => setEmpId(e.target.value)}
                        required
                    />

                </div>

                <div>

                    <label>
                        Name
                    </label>

                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                    />

                </div>

                <div>

                    <label>
                        Role
                    </label>

                    <select
                        value={role}
                        onChange={(e) => setRole(e.target.value)}
                    >

                        <option value="employee">
                            Employee
                        </option>

                        <option value="finance">
                            Finance
                        </option>

                        <option value="engineering">
                            Engineering
                        </option>

                        <option value="marketing">
                            Marketing
                        </option>

                        <option value="hr">
                            HR
                        </option>

                    </select>

                </div>

                {/* Disable the button while the create user request is
                    being processed to prevent duplicate submissions. */}
                <button
                    type="submit"
                    disabled={submitting}
                >

                    {submitting
                        ? "Creating User..."
                        : "Create User"}

                </button>

            </form>

            {message && (
                <p>{message}</p>
            )}

            {error && (
                <p>{error}</p>
            )}

        </main>

    );

}

export default CreateUser;