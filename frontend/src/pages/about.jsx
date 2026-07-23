/**
 * About — Comprehensive project overview page
 *
 * Covers: hero, problem, solution, how it works flow,
 * tech stack, security, RAG explanation, use cases, and highlights.
 */
import { useNavigate } from "react-router-dom";
import PageContainer from "../components/layout/PageContainer";
import Card from "../components/ui/Card";
import Button from "../components/ui/Button";

const techStack = {
    Backend: ["FastAPI", "PostgreSQL", "Psycopg2 Pool", "Argon2 Hashing"],
    Frontend: ["React", "Vite", "React Router", "Axios", "Context API"],
    Database: ["PostgreSQL", "ChromaDB", "Vector Indexes"],
    AI: ["LangChain", "Groq API", "Qwen 3.6 27B", "HuggingFace Embeddings"],
    Authentication: ["JWT", "Refresh Tokens", "HttpOnly Cookies", "Session Binding"],
};

const securityFeatures = [
    { icon: "🔑", title: "Password Hashing", description: "Argon2id hashing protects all stored passwords." },
    { icon: "🎫", title: "JWT Access Tokens", description: "Short-lived tokens authenticate every protected request." },
    { icon: "🔄", title: "Refresh Tokens", description: "Rotated on every refresh; stored only as SHA-256 hashes." },
    { icon: "🛡️", title: "RBAC", description: "Role-based access enforced before document retrieval." },
    { icon: "🔒", title: "Protected APIs", description: "Every endpoint validates session before processing." },
    { icon: "⏱️", title: "Session Expiry", description: "Absolute session limits prevent indefinite access." },
];

const useCases = [
    { icon: "👥", title: "HR", description: "Query employee policies, leave entitlements, and onboarding procedures." },
    { icon: "💰", title: "Finance", description: "Access quarterly reports, financial summaries, and budget data." },
    { icon: "⚙️", title: "Engineering", description: "Search architecture docs, deployment guides, and tech standards." },
    { icon: "📢", title: "Marketing", description: "Review campaign reports, market analysis, and brand guidelines." },
    { icon: "🏛️", title: "Leadership", description: "Organization-wide access to all departmental knowledge." },
];

const highlights = [
    "Enterprise Authentication",
    "Role-Based Access Control",
    "Secure Document Retrieval",
    "Conversational AI",
    "Modern React UI",
    "Responsive Design",
    "Session Management",
    "Vector Search (MMR)",
];

const flowSteps = [
    "Login",
    "Verify Identity",
    "Determine User Role",
    "Retrieve Authorized Documents",
    "Generate AI Response",
    "Deliver Answer",
];


function About() {
    const navigate = useNavigate();

    return (
        <>
            {/* Hero */}
<section className="hero">
    <span className="hero__badge">
        Enterprise AI Platform
    </span>

    <h1>About Novaris</h1>

    <p className="hero__subtitle">
        Secure, role-aware AI assistant built for organizations using
        Retrieval-Augmented Generation, Role-Based Access Control,
        and enterprise-grade authentication.
    </p>

    <div className="hero__actions">
        <Button
            variant="primary"
            size="lg"
            onClick={() => navigate("/demo")}
        >
            Explore Demo
        </Button>

        <Button
            variant="secondary"
            size="lg"
            onClick={() => navigate("/")}
        >
            Back Home
        </Button>
    </div>
</section>

            <PageContainer>

<section className="section">
    <div className="card-grid">

        <Card
            icon="⚠️"
            title="The Problem"
            description="Organizations accumulate thousands of internal documents across HR, Finance, Engineering and other departments. Finding accurate information quickly while maintaining security becomes increasingly difficult."
        />

        <Card
            icon="💡"
            title="The Solution"
            description="Novaris combines Role-Based Access Control with Retrieval-Augmented Generation, allowing employees to ask natural-language questions while ensuring responses are generated only from documents they are authorized to access."
        />

    </div>
</section>

                {/* How Novaris Works — horizontal flow */}
                <section className="section">
                    <div className="section-header">
                        <h2>How Novaris Works</h2>
                    </div>
                    <div className="about-flow">
                        {flowSteps.map((step, index) => (
                            <span key={step} style={{ display: "contents" }}>
                                <span className="about-flow__step">{step}</span>
                                {index < flowSteps.length - 1 && (
                                    <span className="about-flow__arrow">→</span>
                                )}
                            </span>
                        ))}
                    </div>
                </section>

                {/* Technology Stack */}
                <section className="section">
                    <div className="section-header">
                        <h2>Technology Stack</h2>
                    </div>
                    <div className="card-grid">
                        {Object.entries(techStack).map(([category, items]) => (
                            <Card key={category} title={category}>
                                <ul style={{ listStyle: "none", marginTop: "0.75rem" }}>
                                    {items.map((item) => (
                                        <li
                                            key={item}
                                            style={{
                                                padding: "0.35rem 0",
                                                fontSize: "0.875rem",
                                                color: "var(--color-text-secondary)",
                                                borderBottom: "1px solid var(--color-border)",
                                            }}
                                        >
                                            {item}
                                        </li>
                                    ))}
                                </ul>
                            </Card>
                        ))}
                    </div>
                </section>

                {/* Security */}
                <section className="section">
                    <div className="section-header">
                        <h2>Security</h2>
                        <p>Enterprise-grade security that differentiates Novaris from typical student projects.</p>
                    </div>
                    <div className="card-grid">
                        {securityFeatures.map((feature) => (
                            <Card
                                key={feature.title}
                                icon={feature.icon}
                                title={feature.title}
                                description={feature.description}
                            />
                        ))}
                    </div>
                </section>

                {/* Why RAG? */}
                <section className="section">
                    <Card glass hover={false}>
                        <h3 style={{ marginBottom: "1rem" }}>Why RAG?</h3>
                        <p>
                            Retrieval-Augmented Generation (RAG) combines the power of large language
                            models with your organization's private documents. Instead of relying on
                            the model's general training data, RAG first retrieves relevant document
                            chunks and then generates answers grounded in that specific context.
                        </p>
                        <p style={{ marginTop: "1rem" }}>
                            This means Novaris can answer questions about your company's quarterly
                            financial reports, HR policies, or engineering standards — with answers
                            sourced directly from authorized documents, not hallucinated from public data.
                        </p>
                    </Card>
                </section>

                {/* Use Cases */}
                <section className="section">
                    <div className="section-header">
                        <h2>Use Cases</h2>
                    </div>
                    <div className="card-grid">
                        {useCases.map((useCase) => (
                            <Card
                                key={useCase.title}
                                icon={useCase.icon}
                                title={useCase.title}
                                description={useCase.description}
                            />
                        ))}
                    </div>
                </section>

                {/* Project Highlights */}
                <section className="section">
                    <div className="section-header">
                        <h2>Project Highlights</h2>
                    </div>
                    <div className="card-grid">
                        {highlights.map((highlight) => (
                            <Card key={highlight} title={highlight} hover={false} />
                        ))}
                    </div>
                </section>

                <section className="section text-center">
                    <Button variant="primary" size="lg" onClick={() => navigate("/demo")}>
                        Explore the Demo
                    </Button>
                </section>
            </PageContainer>
        </>
    );
}

export default About;
