/**
 * ChangePassword — Authenticated password change
 *
 * Preserves session verification, password validation, and automatic
 * logout after successful change (matching backend security behavior).
 */
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { changePassword, logout } from "../services/auth";
import { useAuth } from "../contexts/AuthContext";
import useVerifySession from "../hooks/use_verify_session";
import AuthCard from "../components/layout/AuthCard";
import Button from "../components/ui/Button";
import LoadingSpinner from "../components/ui/LoadingSpinner";

function ChangePassword() {
    const navigate = useNavigate();
    const { user, setUser, loading } = useAuth();
    const verifySession = useVerifySession();
    const [checking, setChecking] = useState(true);

    const [empId, setEmpId] = useState("");
    const [oldPassword, setOldPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

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

    async function handleSubmit(event) {
        event.preventDefault();
        setError("");
        setSuccess("");

        if (!empId.trim()) { setError("Employee ID is required."); return; }
        if (!oldPassword.trim()) { setError("Current password is required."); return; }
        if (!newPassword.trim()) { setError("New password is required."); return; }
        if (!confirmPassword.trim()) { setError("Please confirm your new password."); return; }
        if (newPassword !== confirmPassword) { setError("New passwords do not match."); return; }

        if (empId.trim().toLowerCase() !== user?.emp_id?.toLowerCase()) {
            setError("You can only change your own password.");
            return;
        }

        setSubmitting(true);

        try {
            await changePassword(empId.trim(), oldPassword, newPassword);
            setSuccess("Password changed successfully. Logging out...");

            try {
                await logout();
            } catch (err) {
                console.error("Backend logout failed:", err);
            } finally {
                sessionStorage.removeItem(`chat_history_${empId.trim()}`);
                setUser(null);
                navigate("/login");
            }
        } catch (err) {
            setError(err.response?.data?.detail || "Something went wrong.");
        } finally {
            setSubmitting(false);
        }
    }

    if (loading || checking) {
        return <LoadingSpinner text="Verifying session..." />;
    }

    return (
        <AuthCard
            title="Change Password"
            subtitle="After a successful change, you will be logged out and must sign in with your new password."
        >
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label className="form-label">Employee ID</label>
                    <input
                        className="form-input"
                        type="text"
                        value={empId}
                        onChange={(event) => setEmpId(event.target.value)}
                        required
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Current Password</label>
                    <input
                        className="form-input"
                        type="password"
                        value={oldPassword}
                        onChange={(event) => setOldPassword(event.target.value)}
                        required
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">New Password</label>
                    <input
                        className="form-input"
                        type="password"
                        value={newPassword}
                        onChange={(event) => setNewPassword(event.target.value)}
                        required
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Confirm New Password</label>
                    <input
                        className="form-input"
                        type="password"
                        value={confirmPassword}
                        onChange={(event) => setConfirmPassword(event.target.value)}
                        required
                    />
                </div>

                {error && <p className="form-error">{error}</p>}
                {success && <p className="form-success">{success}</p>}

                <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    disabled={submitting}
                    style={{ width: "100%" }}
                >
                    {submitting ? "Changing Password..." : "Change Password"}
                </Button>
            </form>
        </AuthCard>
    );
}

export default ChangePassword;
