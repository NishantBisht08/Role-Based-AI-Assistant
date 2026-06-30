import { useAuth } from "../contexts/AuthContext";
import { Link } from "react-router-dom";


const folderDisplayNames = {
    engineering: "Engineering Documents",
    finance: "Financial Reports",
    hr: "HR Policies",
    marketing: "Marketing Resources",
    general: "Employee Handbook",
};



function Dashboard() {

    const { user, loading } = useAuth();

    if (loading) {
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