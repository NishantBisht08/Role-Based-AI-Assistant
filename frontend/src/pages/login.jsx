import { useState } from "react";
import { Link } from "react-router-dom";
import { login, getCurrentUser } from "../services/auth";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";


function Login() {

    const navigate = useNavigate();
    const { setUser } = useAuth();

    // Stores the Employee ID entered by the user
    const [empId, setEmpId] = useState("");

    // Stores the Password entered by the user
    const [password, setPassword] = useState("");

    // Tracks whether a login request is in progress
    const [loading, setLoading] = useState(false);

    // Stores any error message returned during login
    const [error, setError] = useState("");



    const handleLogin = async (event) => {

    // Prevents the browser from refreshing the page
    event.preventDefault();

    // Show loading state and clear previous errors
    setLoading(true);
    setError("");

    try {

        // 1. Authenticate the user
        await login(empId, password);

        const currentUser = await getCurrentUser();

        setUser(currentUser);

        navigate("/dashboard");

    }

catch (err) {

    console.log(err);

    console.log(err.response);

    console.log(err.message);

    setError(

        err.response?.data?.detail ||

        "Something went wrong."

    );

}

    finally {

        setLoading(false);

    }

};

    return (

        <main>

            <section>

                <h2>Login</h2>

                <p>
                    Sign in using your Employee ID and password.
                </p>

                <form onSubmit={handleLogin}>

                    <label htmlFor="empId">
                        Employee ID
                    </label>

                    <input
                        type="text"
                        id="empId"
                        value={empId}
                        onChange={(event) => setEmpId(event.target.value)}
                        placeholder="Enter Employee ID"
                    />

                    <label htmlFor="password">
                        Password
                    </label>

                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(event) => setPassword(event.target.value)}
                        placeholder="Enter Password"
                    />

                    {
                        error && (

                            <p>
                                {error}
                            </p>

                        )
                    }

                    <button
                        type="submit"
                        disabled={loading}
                    >

                        {
                            loading
                                ? "Logging In..."
                                : "Login"
                        }

                    </button>

                </form>

                <p>

                    First time here?

                    {" "}

                    <Link to="/set-password">
                        Set Password
                    </Link>

                </p>

            </section>

        </main>

    );

}

export default Login;