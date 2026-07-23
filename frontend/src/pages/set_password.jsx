/**
 * SetPassword — First-time password initialization
 *
 * Uses the shared AuthCard layout. Preserves existing validation
 * and setPassword() API integration without backend changes.
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { setPassword } from "../services/auth";
import AuthCard from "../components/layout/AuthCard";
import Button from "../components/ui/Button";

function SetPassword() {
    const navigate = useNavigate();

    const [empId, setEmpId] = useState("");
    const [password, setPasswordValue] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

    async function handleSubmit(event) {
        event.preventDefault();
        setError("");
        setSuccess("");

        if (!empId.trim()) {
            setError("Employee ID is required.");
            return;
        }
        if (!password.trim()) {
            setError("Password is required.");
            return;
        }
        if (!confirmPassword.trim()) {
            setError("Please confirm your password.");
            return;
        }
        if (password !== confirmPassword) {
            setError("Passwords do not match.");
            return;
        }

        setLoading(true);

        try {
            await setPassword(empId.trim(), password);
            setSuccess("Password set successfully. Redirecting to login...");
            setTimeout(() => navigate("/login"), 1000);
        } catch (err) {
            setError(err.response?.data?.detail || "Something went wrong.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <AuthCard
            title="Set Password"
            subtitle="Initialize your account password for first-time access."
            footer={
                <>
                    Already have a password?{" "}
                    <Button variant="ghost" size="sm" onClick={() => navigate("/login")}>
                        Login
                    </Button>
                </>
            }
        >
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label className="form-label">Employee ID</label>
                    <input
                        className="form-input"
                        type="text"
                        value={empId}
                        onChange={(event) => setEmpId(event.target.value)}
                        placeholder="Enter Employee ID"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">New Password</label>
                    <input
                        className="form-input"
                        type="password"
                        value={password}
                        onChange={(event) => setPasswordValue(event.target.value)}
                        placeholder="Create a password"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Confirm Password</label>
                    <input
                        className="form-input"
                        type="password"
                        value={confirmPassword}
                        onChange={(event) => setConfirmPassword(event.target.value)}
                        placeholder="Confirm your password"
                    />
                </div>

                {error && <p className="form-error">{error}</p>}
                {success && <p className="form-success">{success}</p>}

                <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    disabled={loading}
                    style={{ width: "100%" }}
                >
                    {loading ? "Setting Password..." : "Set Password"}
                </Button>
            </form>
        </AuthCard>
    );
}

export default SetPassword;
