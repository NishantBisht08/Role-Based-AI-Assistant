/**
 * DemoGuide - interview-friendly walkthrough for Novaris
 *
 * This page is intentionally static: it helps evaluators explore the
 * already-implemented authentication, RBAC dataset, and chat flows without
 * touching backend contracts or application state.
 */
import PageContainer from "../components/layout/PageContainer";
import Card from "../components/ui/Card";
import CopyButton from "../components/common/CopyButton";

const demoAccounts = [
    { role: "Finance", empId: "sid", password: "sid123", note: "Finance reports only" },
    { role: "Engineering", empId: "nishant", password: "nishant123", note: "Engineering documents" },
    { role: "HR", empId: "khushi", password: "khushi123", note: "HR plus handbook" },
    { role: "Marketing", empId: "kavya", password: "kavya123", note: "Marketing reports" },
    { role: "Employee", empId: "arjun", password: "arjun123", note: "General handbook" },
    { role: "C-Level", empId: "leo", password: "leo123", note: "Organization-wide documents" },
];

const recommendedFlow = [
    "Login as a department user such as Finance.",
    "Open the protected Dataset page and observe the limited document set.",
    "Start NovaMind and ask a department-specific question.",
    "Logout, then login as another department to compare access boundaries.",
    "Login as C-Level to observe organization-wide document visibility.",
];

function DemoGuide() {
    return (
        <PageContainer>
            <section className="hero" style={{ marginLeft: "calc(50% - 50vw)", marginRight: "calc(50% - 50vw)" }}>
                <span className="hero__badge">Guided Product Tour</span>
                <h1>How to Explore Novaris</h1>
                <p className="hero__subtitle">
                    Novaris is an Enterprise Knowledge Platform. Use this guide to
                    demonstrate secure login, role-based document access, and AI answers
                    grounded in authorized company knowledge.
                </p>
            </section>

            <section className="section">
                <div className="section-header">
                    <h2>Best Demo Experience</h2>
                    <p>
                        Try multiple accounts to see how the same product adapts to each
                        employee role without changing the backend or chat workflow.
                    </p>
                </div>
                <div className="card-grid">
                    <Card icon="1" title="Finance" description="Review finance-only access and ask revenue or margin questions." />
                    <Card icon="2" title="Engineering" description="Switch to engineering to inspect technical knowledge boundaries." />
                    <Card icon="3" title="C-Level" description="Finish with broad access to show executive-level visibility." />
                </div>
            </section>

            <section className="section">
                <div className="section-header">
                    <h2>Demo Accounts</h2>
                    <p>Copy credentials directly, then sign in from the Login page.</p>
                </div>
                <div style={{ overflowX: "auto" }}>
                    <table className="demo-table">
                        <thead>
                            <tr>
                                <th>Role</th>
                                <th>Employee ID</th>
                                <th>Password</th>
                                <th>What to Observe</th>
                            </tr>
                        </thead>
                        <tbody>
                            {demoAccounts.map((account) => (
                                <tr key={account.empId}>
                                    <td>{account.role}</td>
                                    <td>
                                        <span className="demo-table__copy-cell">
                                            {account.empId}
                                            <CopyButton text={account.empId} />
                                        </span>
                                    </td>
                                    <td>
                                        <span className="demo-table__copy-cell">
                                            {account.password}
                                            <CopyButton text={account.password} />
                                        </span>
                                    </td>
                                    <td>{account.note}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            <section className="section">
                <div className="section-header">
                    <h2>Recommended Demo Flow</h2>
                    <p>A short path that highlights the strongest Version 1 functionality.</p>
                </div>
                <ol className="demo-steps">
                    {recommendedFlow.map((step) => (
                        <li key={step}>{step}</li>
                    ))}
                </ol>
            </section>
        </PageContainer>
    );
}

export default DemoGuide;
