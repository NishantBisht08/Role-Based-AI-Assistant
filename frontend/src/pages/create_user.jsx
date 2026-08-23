/**
 * CreateUser — Admin-only user creation page
 *
 * Preserves admin role check, session verification, and createUser() API.
 * Uses AuthCard layout for visual consistency with other auth pages.
 */
import { useState} from "react";
import { Navigate, useNavigate } from "react-router-dom";

import { useAuth } from "../contexts/AuthContext";
import { createUser } from "../services/auth";
import AuthCard from "../components/layout/AuthCard";
import Button from "../components/ui/Button";

function CreateUser() {
    const navigate = useNavigate();
    const { user } = useAuth();

    const [empId, setEmpId] = useState("");
    const [name, setName] = useState("");
    const [role, setRole] = useState("employee");
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");
    const [submitting, setSubmitting] = useState(false);

    if (user?.role !== "admin") {
        return <Navigate to="/dashboard" replace />;
    }

    async function handleSubmit(event) {
        event.preventDefault();
        setMessage("");
        setError("");
        setSubmitting(true);

        try {
            const response = await createUser(empId, name, role);
            setMessage(response.message);
            setEmpId("");
            setName("");
            setRole("employee");
        } catch (err) {
            setError(err.response?.data?.detail || "Failed to create user.");
        } finally {
            setSubmitting(false);
        }
    }

    return (
        <AuthCard
            title="Create User"
            subtitle="Add a new employee to the Novaris platform."
            footer={
                <Button variant="ghost" size="sm" onClick={() => navigate("/dashboard")}>
                    Back to Dashboard
                </Button>
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
                        required
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Name</label>
                    <input
                        className="form-input"
                        type="text"
                        value={name}
                        onChange={(event) => setName(event.target.value)}
                        required
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Role</label>
                    <select
                        className="form-select"
                        value={role}
                        onChange={(event) => setRole(event.target.value)}
                    >
                        <option value="employee">Employee</option>
                        <option value="finance">Finance</option>
                        <option value="engineering">Engineering</option>
                        <option value="marketing">Marketing</option>
                        <option value="hr">HR</option>
                    </select>
                </div>

                {message && <p className="form-success">{message}</p>}
                {error && <p className="form-error">{error}</p>}

                <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    disabled={submitting}
                    style={{ width: "100%" }}
                >
                    {submitting ? "Creating User..." : "Create User"}
                </Button>
            </form>
        </AuthCard>
    );
}

export default CreateUser;
