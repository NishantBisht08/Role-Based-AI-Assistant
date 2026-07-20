import { setPassword } from "../services/auth";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

function SetPassword() {

const navigate = useNavigate();

const [empId, setEmpId] = useState("");
const [password, setPasswordValue] = useState("");
const [confirmPassword, setConfirmPassword] = useState("");

const [loading, setLoading] = useState(false);
const [error, setError] = useState("");
const [success, setSuccess] = useState("");

async function handleSubmit(e) {
    e.preventDefault();

    setError("");
    setSuccess("");

    // Validation will go here
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
        setTimeout(() => {
            navigate("/login");
          }, 1000);

    } catch (err) {
        setError(
           err.response?.data?.detail || "Something went wrong."
         );

    } finally {
        setLoading(false);
    }
}

return (
    <div>

        <h1>Set Password</h1>

        <form onSubmit={handleSubmit}>

            <label>Employee ID</label>
            <br />

            <input
                type="text"
                value={empId}
                onChange={(e) => setEmpId(e.target.value)}
            />

            <br /><br />

            <label>New Password</label>
            <br />

            <input
                type="password"
                value={password}
                onChange={(e) => setPasswordValue(e.target.value)}
            />

            <br /><br />

            <label>Confirm Password</label>
            <br />

            <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
            />

            <br /><br />

            <button
                type="submit"
                disabled={loading}
            >
                {loading ? "Setting Password..." : "Set Password"}
            </button>

        </form>

        {error && (
            <p>{error}</p>
        )}

        {success && (
            <p>{success}</p>
        )}

    </div>
);

}

export default SetPassword;
