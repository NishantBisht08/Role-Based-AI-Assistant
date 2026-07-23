/**
 * AboutAuthor — Meet the Builders page
 *
 * Displays both authors with avatars, names, and signature quotes.
 */
import PageContainer from "../components/layout/PageContainer";
import Card from "../components/ui/Card";

const authors = [
    {
        name: "Siddharth Jain",
        initials: "SJ",
        quote: "Believes every bug is just an undocumented feature... until proven otherwise.",
    },
    {
        name: "Nishant Bisht",
        initials: "NB",
        quote: "Specializes in turning coffee into code and impossible ideas into weekend projects.",
    },
];

function AboutAuthor() {
    return (
        <>
            <section className="hero" style={{ paddingBottom: "3rem" }}>
                <h1>Meet the Builders</h1>
                <p className="hero__subtitle">
                    Two computer science students, countless debugging sessions, and an
                    unhealthy number of &ldquo;It worked on my machine&rdquo; moments.
                </p>
            </section>

            <PageContainer narrow>
                <div className="author-grid">
                    {authors.map((author) => (
                        <Card key={author.name} hover={false} className="author-card">
                            <div className="author-card__avatar">{author.initials}</div>
                            <h3>{author.name}</h3>
                            <p className="author-card__quote">&ldquo;{author.quote}&rdquo;</p>
                        </Card>
                    ))}
                </div>

                <div className="section text-center" style={{ marginTop: "3rem" }}>
                    <Card glass hover={false}>
                        <p style={{ fontSize: "1.05rem", lineHeight: 1.8 }}>
                            Built by <strong>Siddharth Jain</strong> &amp; <strong>Nishant Bisht</strong>
                            <br />
                            <span className="text-muted">
                                Novaris — where enterprise security meets intelligent search.
                            </span>
                        </p>
                    </Card>
                </div>
            </PageContainer>
        </>
    );
}

export default AboutAuthor;
