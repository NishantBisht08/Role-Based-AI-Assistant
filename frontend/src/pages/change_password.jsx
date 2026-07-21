import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { changePassword, logout } from "../services/auth";
import { useAuth } from "../contexts/AuthContext";
import useVerifySession from "../hooks/use_verify_session";

function ChangePassword() {

    const navigate = useNavigate();

    // Currently authenticated user
    const { user, setUser, loading } = useAuth();

    // Hook used to verify whether the current session is still valid
    const verifySession = useVerifySession();

    // Prevent page rendering until session verification finishes
    const [checking, setChecking] = useState(true);

    const [empId, setEmpId] = useState("");
    const [oldPassword, setOldPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");

    // Tracks whether the password change request is in progress
    const [submitting, setSubmitting] = useState(false);

    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

    // Verify the session every time this protected page is opened.
    // If the refresh token has expired, redirect the user to login
    // before showing the form.
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

    async function handleSubmit(e) {

        e.preventDefault();

        setError("");
        setSuccess("");

        if (!empId.trim()) {
            setError("Employee ID is required.");
            return;
        }

        if (!oldPassword.trim()) {
            setError("Current password is required.");
            return;
        }

        if (!newPassword.trim()) {
            setError("New password is required.");
            return;
        }

        if (!confirmPassword.trim()) {
            setError("Please confirm your new password.");
            return;
        }

        if (newPassword !== confirmPassword) {
            setError("New passwords do not match.");
            return;
        }

        // Prevent users from attempting to change
        // another employee's password.
        if (empId.trim().toLowerCase() !== user?.emp_id?.toLowerCase()) {

            setError("You can only change your own password.");

            return;

        }

        setSubmitting(true);

        try {

            await changePassword(
                empId.trim(),
                oldPassword,
                newPassword
            );

            setSuccess("Password changed successfully. Logging out...");

            try {

                // Invalidate refresh token on the backend
                await logout();

            } catch (err) {

                console.error("Backend logout failed:", err);

            } finally {

                // Remove locally stored chat history
                sessionStorage.removeItem(
                    `chat_history_${empId.trim()}`
                );

                // Clear authenticated user
                setUser(null);

                // Redirect back to login
                navigate("/login");

            }

        } catch (err) {

            setError(
                err.response?.data?.detail ||
                "Something went wrong."
            );

        } finally {

            setSubmitting(false);

        }

    }

    // Wait until both AuthContext and session verification finish
    // before rendering the page.
    if (loading || checking) {

        return <h2>Loading...</h2>;

    }

    return (

        <div>

            <h2>Change Password</h2>

            <p>
                After a successful password change, you will be automatically logged out and asked to sign in again using your new password.
            </p>

            <form onSubmit={handleSubmit}>

                <label>Employee ID</label>

                <input
                    type="text"
                    value={empId}
                    onChange={(e) => setEmpId(e.target.value)}
                    required
                />

                <label>Current Password</label>

                <input
                    type="password"
                    value={oldPassword}
                    onChange={(e) => setOldPassword(e.target.value)}
                    required
                />

                <label>New Password</label>

                <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                />

                <label>Confirm New Password</label>

                <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                />

                <button
                    type="submit"
                    disabled={submitting}
                >
                    {submitting
                        ? "Changing Password..."
                        : "Change Password"}
                </button>

            </form>

            {error && <p>{error}</p>}

            {success && <p>{success}</p>}

        </div>

    );

}

export default ChangePassword;