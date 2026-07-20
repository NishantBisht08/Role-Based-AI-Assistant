import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { createUser } from "../services/auth";

function CreateUser() {

    const { user } = useAuth();

    const [empId, setEmpId] = useState("");
    const [name, setName] = useState("");
    const [role, setRole] = useState("employee");

    const [message, setMessage] = useState("");
    const [error, setError] = useState("");

    if (user.role !== "admin") {
        return <Navigate to="/dashboard" replace />;
    }

    async function handleSubmit(e) {

        e.preventDefault();

        setMessage("");
        setError("");

        try {

            const response = await createUser(
                empId,
                name,
                role
            );

            setMessage(response.message);

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

                <button type="submit">
                    Create User
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