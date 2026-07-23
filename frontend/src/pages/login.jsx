/**
 * Login — Authentication page
 *
 * Centered glassmorphism card for Employee ID + password login.
 * Preserves existing login flow: login() → getCurrentUser() → setUser → navigate.
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, getCurrentUser } from "../services/auth";
import { useAuth } from "../contexts/AuthContext";
import AuthCard from "../components/layout/AuthCard";
import Button from "../components/ui/Button";

function Login() {
    const navigate = useNavigate();
    const { setUser } = useAuth();

    const [empId, setEmpId] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleLogin = async (event) => {
        event.preventDefault();
        setLoading(true);
        setError("");

        try {
            await login(empId, password);
            const currentUser = await getCurrentUser();
            setUser(currentUser);
            navigate("/dashboard");
        } catch (err) {
            setError(
                err.response?.data?.detail || "Something went wrong."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <AuthCard
            title="Welcome Back"
            subtitle="Sign in to Novaris using your Employee ID and password."
            footer={
                <>
                    First time here?{" "}
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigate("/set-password")}
                    >
                        Set Password
                    </Button>
                </>
            }
        >
            <form onSubmit={handleLogin}>
                <div className="form-group">
                    <label className="form-label" htmlFor="empId">
                        Employee ID
                    </label>
                    <input
                        className="form-input"
                        type="text"
                        id="empId"
                        value={empId}
                        onChange={(event) => setEmpId(event.target.value)}
                        placeholder="Enter Employee ID"
                        autoComplete="username"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="password">
                        Password
                    </label>
                    <input
                        className="form-input"
                        type="password"
                        id="password"
                        value={password}
                        onChange={(event) => setPassword(event.target.value)}
                        placeholder="Enter Password"
                        autoComplete="current-password"
                    />
                </div>

                {error && <p className="form-error">{error}</p>}

                <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    disabled={loading}
                    style={{ width: "100%" }}
                >
                    {loading ? "Logging In..." : "Login"}
                </Button>
            </form>
        </AuthCard>
    );
}

export default Login;
