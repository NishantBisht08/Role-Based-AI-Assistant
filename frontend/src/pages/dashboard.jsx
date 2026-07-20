import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import useVerifySession from "../hooks/use_verify_session";

const folderDisplayNames = {
    engineering: "Engineering Documents",
    finance: "Financial Reports",
    hr: "HR Policies",
    marketing: "Marketing Resources",
    general: "Employee Handbook",
};



function Dashboard() {

    const navigate = useNavigate();
    const verifySession = useVerifySession();
    const [checking, setChecking] = useState(true);
    
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

}, [navigate]);


    const { user, loading } = useAuth();

    if (loading || checking) {
        return <h2>Loading...</h2>;
    }

    return (

        <main>

 <h2>Dashboard</h2>

            {/* Welcome Card */}

            <section>

                <h3>
                    Welcome back, {user.name} 👋
                </h3>

                <p>
                    Employee ID
                </p>

                <p>
                    {user.emp_id.toUpperCase()}
                </p>

                <p>
                    Role
                </p>

                <p>
                    {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                </p>

            </section>


            {/* AI Assistant */}

            <section>

                <h3>
                    AI Assistant
                </h3>

                <p>
                    Ask questions about company documents using our AI assistant.
                </p>

                <Link to="/chat">

                    <button>
                        Start Chat
                    </button>

                </Link>

            </section>


            {/* Quick Actions */}

            <section>

                <h3>
                    Quick Actions
                </h3>

                <div>

                    <Link to="/dataset">

                        <button>
                            View Dataset
                        </button>

                    </Link>

                    <Link to="/change-password">

                        <button>
                            Change Password
                        </button>

                    </Link>

                    {user.role === "admin" && (     //Shows the create user button if your role is admin, this is like the if condition, called conditional rendering in react

                    <Link to="/create-user">

                        <button>
                            Create User
                        </button>

                    </Link>

                     )}

                </div>

            </section>


            {/* About Your Role */}

            <section>

                <h3>
                    About Your Role
                </h3>

<ul>

    {user.accessible_folders.map((folder) => (

        <li key={folder}>
            {folderDisplayNames[folder]}
        </li>

    ))}

</ul>

            </section>

        </main>

    );
}

export default Dashboard;