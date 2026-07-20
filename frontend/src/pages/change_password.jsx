import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { changePassword, logout } from "../services/auth";
import { useAuth } from "../contexts/AuthContext";

function ChangePassword() {

    const navigate = useNavigate();

    const { user, setUser } = useAuth();

    const [empId, setEmpId] = useState("");
    const [oldPassword, setOldPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");

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

    if (empId.trim().toLowerCase() !== user?.emp_id?.toLowerCase()) {
    setError("You can only change your own password.");
    return;
}

    setLoading(true);

    try {

    await changePassword(
    empId.trim(),
    oldPassword,
    newPassword
);

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


    }
    
    catch (err) {

        setError(
            err.response?.data?.detail || "Something went wrong."
        );

    } finally {

        setLoading(false);

    }
}


return (
    // Your form JSX here

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

            <button type="submit" disabled={loading}>

                {loading ? "Changing Password..." : "Change Password"}

            </button>

        </form>

        {error && <p>{error}</p>}

        {success && <p>{success}</p>}

    </div>

);


}

export default ChangePassword;

