/**
 * Home — Novaris landing page
 *
 * Communicates that Novaris is an Enterprise Knowledge Platform.
 * Features: hero, feature cards (max 3/row), demo gate entrance,
 * "Did You Know?" flip cards, and navigation buttons.
 */

import aiImage from "../assets/images/novaris.png";
import { useNavigate } from "react-router-dom";
import PageContainer from "../components/layout/PageContainer";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";
import FlipCard from "../components/common/FlipCard";
import GateEntrance from "../components/common/GateEntrance";

const features = [
    {
        icon: "🔐",
        title: "JWT Authentication",
        description: "Secure login using HttpOnly cookies with access and refresh tokens.",
    },
    {
        icon: "🛡️",
        title: "Role-Based Access",
        description: "Users can only access documents permitted by their assigned role.",
    },
    {
        icon: "🤖",
        title: "AI-Powered RAG",
        description: "Answers are generated only from authorized documents via semantic search.",
    },
    {
        icon: "⚡",
        title: "FastAPI Backend",
        description: "High-performance backend with modular, enterprise-grade architecture.",
    },
    {
        icon: "🗄️",
        title: "PostgreSQL Database",
        description: "Secure storage for users, sessions, roles, and refresh token hashes.",
    },
    {
        icon: "🔍",
        title: "LangChain + ChromaDB",
        description: "Retrieves relevant document chunks using vector similarity search.",
    },
];

const reasons = [
    {
        icon: "🎯",
        title: "Role-Aware Responses",
        description: "Answers are generated according to the permissions assigned to your role.",
    },
    {
        icon: "💬",
        title: "Contextual Q&A",
        description: "Uses Retrieval-Augmented Generation to answer from company documents.",
    },
    {
        icon: "🔒",
        title: "Secure Sessions",
        description: "Cookie-based authentication with automatic token refresh and session expiry.",
    },
    {
        icon: "🏢",
        title: "Enterprise Architecture",
        description: "Modular FastAPI services with scalable PostgreSQL and ChromaDB support.",
    },
    {
        icon: "⚡",
        title: "Fast Retrieval",
        description: "Relevant documents are retrieved via MMR before generating AI responses.",
    },
    {
        icon: "👤",
        title: "Admin Control",
        description: "Administrators can create users and manage organizational access.",
    },
];

const facts = [
    {
        title: "Two users can ask the same question and receive different answers.",
        description: "Responses depend on each user's assigned permissions.",
    },
    {
        title: "Vector search finds meaning, not just matching words.",
        description: "Similar ideas can be found even without identical keywords.",
    },
    {
        title: "Access tokens have short lifetimes.",
        description: "Short-lived access tokens reduce the risk of unauthorized access.",
    },
    {
        title: "The AI can't retrieve what you're not allowed to see.",
        description: "Access control is enforced before retrieval begins.",
    },
];

function Home() {
    const navigate = useNavigate();

    return (
        <>
            {/* Hero — immediately communicates enterprise positioning */}
            <section
    className="hero"
    style={{
backgroundImage: `
linear-gradient(
    rgba(255,255,255,0.05),
    rgba(15,23,42,0.15)
),
url(${aiImage})
`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
    }}
>
                <span className="hero__badge"><br></br></span>
                <h1><br></br></h1>
                <p className="hero__subtitle">
                    <br></br>
                     <br></br>
                </p>
<div className="hero__actions">
    <Button
        variant="primary"
        size="lg"
        onClick={() => navigate("/login")}
    >
        Get Started
    </Button>

    <Button
        variant="secondary"
        size="lg"
        onClick={() => navigate("/about")}
    >
        Learn More
    </Button>
</div>
            </section>

            <PageContainer>
                {/* Features — max 3 cards per row via card-grid CSS */}
                <section className="section">
                    <div className="section-header">
                        <h2>Platform Features</h2>
                        <p>Built with enterprise-grade security and AI at its core.</p>
                    </div>
                    <div className="card-grid">
                        {features.map((feature) => (
                            <Card
                                key={feature.title}
                                icon={feature.icon}
                                title={feature.title}
                                description={feature.description}
                            />
                        ))}
                    </div>
                </section>

                {/* Demo Guide — gate entrance interaction */}
                <section className="section text-center">
                    <div className="section-header">
                        <h2>Demo Guide</h2>
                        <p>
                            New here? Explore the application through a step-by-step
                            walkthrough before logging in.
                        </p>
                    </div>
                    <GateEntrance />
                </section>

                {/* Why Choose Novaris */}
                <section className="section">
                    <div className="section-header">
                        <h2>Why Choose Novaris?</h2>
                        <p>Purpose-built for organizations that take knowledge security seriously.</p>
                    </div>
                    <div className="card-grid">
                        {reasons.map((reason) => (
                            <Card
                                key={reason.title}
                                icon={reason.icon}
                                title={reason.title}
                                description={reason.description}
                            />
                        ))}
                    </div>
                </section>

                {/* Dataset */}
                <section className="section text-center">
                    <div className="section-header">
                        <h2>Dataset</h2>
                        <p>Explore the documents and datasets that power Novaris.</p>
                    </div>
                    <Button variant="primary" size="lg" onClick={() => navigate("/dataset")}>
                        View Dataset
                    </Button>
                </section>

                {/* Did You Know? — 3D flip cards */}
                <section className="section">
                    <div className="section-header">
                        <h2>Did You Know?</h2>
                        <p>Click a card to reveal an interesting fact about Novaris.</p>
                    </div>
                    <div className="card-grid">
                        {facts.map((fact) => (
                            <FlipCard
                                key={fact.title}
                                backTitle={fact.title}
                                backDescription={fact.description}
                            />
                        ))}
                    </div>
                </section>

                {/* Authors */}
                <section className="section text-center">
                    <div className="section-header">
                        <h2>Meet the Builders</h2>
                        <p>Learn more about the team behind Novaris.</p>
                    </div>
                    <Button variant="secondary" size="lg" onClick={() => navigate("/about-author")}>
                        About the Authors
                    </Button>
                </section>
            </PageContainer>
        </>
    );
}

export default Home;
