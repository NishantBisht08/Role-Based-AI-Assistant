/**
 * Dashboard — Authenticated user home
 *
 * Displays welcome card, NovaMind AI section, quick actions,
 * and role-based access level card. Preserves all existing
 * session verification and RBAC display logic.
 */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import useVerifySession from "../hooks/use_verify_session";
import PageContainer from "../components/layout/PageContainer";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import LoadingSpinner from "../components/ui/LoadingSpinner";
import novaImage from "../assets/images/nova.png";

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
    }, [navigate, verifySession]);

    const { user, loading } = useAuth();

    if (loading || checking) {
        return <LoadingSpinner text="Loading dashboard..." />;
    }

    // Avatar initials from user name
    const initials = user.name
        .split(" ")
        .map((part) => part[0])
        .join("")
        .toUpperCase()
        .slice(0, 2);

return (
    <div
        style={{
            backgroundImage: `
                linear-gradient(
         rgba(6,12,22,0.82),
    rgba(6,12,22,0.86)
                ),
                url(${novaImage})
            `,
            backgroundSize: "cover",
            backgroundPosition: "center",
            backgroundRepeat: "no-repeat",
            backgroundAttachment: "fixed",
            minHeight: "100vh",
        }}
    >
        <PageContainer>
            {/* Welcome Card — centered with avatar and employee details */}
            <section className="dashboard-welcome animate-fade-in-up">
                <div className="dashboard-avatar">{initials}</div>
                <h2>
                    Welcome back, {user.name} 👋
                </h2>
                <div className="dashboard-meta">
                    <div className="dashboard-meta__item">
                        <div className="dashboard-meta__label">Employee ID</div>
                        <div className="dashboard-meta__value">
                            {user.emp_id.toUpperCase()}
                        </div>
                    </div>
                    <div className="dashboard-meta__item">
                        <div className="dashboard-meta__label">Role</div>
                        <div className="dashboard-meta__value">
                            {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                        </div>
                    </div>
                </div>
            </section>

            {/* NovaMind AI Assistant */}
            <section className="dashboard-section">
                <div className="dashboard-ai-card">
                    <h3>🧠 NovaMind</h3>
                    <p>
                        Ask questions about company documents using AI.
                        Your role determines which documents NovaMind can access.
                    </p>
                    <Button variant="primary" size="lg" onClick={() => navigate("/chat")}>
                        Start Chat
                    </Button>
                </div>
            </section>

            {/* Quick Actions */}
            <section className="dashboard-section">
                <h3>Quick Actions</h3>
                <div className="dashboard-actions">
                    <Button variant="secondary" onClick={() => navigate("/dashboard/dataset")}>
                        📄 View Dataset
                    </Button>
                    <Button variant="secondary" onClick={() => navigate("/change-password")}>
                        🔑 Change Password
                    </Button>
                    {user.role === "admin" && (
                        <Button variant="secondary" onClick={() => navigate("/create-user")}>
                            👤 Create User
                        </Button>
                    )}
                </div>
            </section>

            {/* Access Level Card — shows RBAC permissions */}
            <section className="dashboard-section">
                <Card hover={false}>
                    <div className="access-card">
                        <div className="access-card__icon">🔒</div>
                        <div>
                            <h3>Access Level</h3>
                            <p className="text-muted" style={{ marginBottom: "0.5rem" }}>
                                Role: <strong>{user.role.charAt(0).toUpperCase() + user.role.slice(1)}</strong>
                            </p>
                            {user.accessible_folders.length > 0 ? (
                                <ul className="access-card__folders">
                                    {user.accessible_folders.map((folder) => (
                                        <li key={folder}>
                                            {folderDisplayNames[folder] || folder}
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-muted">
                                    No document folders accessible for this role.
                                </p>
                            )}
                            <span className="access-card__badge">
                                ✓ Role-Based Access is Active
                            </span>
                        </div>
                    </div>
                </Card>
            </section>
        </PageContainer>
        </div>
    );
}

export default Dashboard;
