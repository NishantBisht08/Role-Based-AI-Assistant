/**
 * Footer — Playful footer with hover easter eggs
 *
 * Three hover interactions:
 *  - Brand text → "Beware! AI is listening 👀"
 *  - Built with → "Built with coffee and curiosity ☕"
 *  - Copyright → "Every great answer begins with a great question."
 */
function Footer() {
    return (
        <footer className="footer">
            <div className="footer__inner">
                {/* Easter egg: brand hover */}
                <div className="footer__brand">
                    <span className="footer__default">Novaris</span>
                    <span className="footer__hover">Beware! AI is listening 👀</span>
                </div>

                {/* Easter egg: tech stack hover */}
                <div className="footer__text">
                    <span className="footer__default">
                        Built with React • FastAPI • LangChain
                    </span>
                    <span className="footer__hover">
                        Built with coffee and curiosity ☕
                    </span>
                </div>

                {/* Easter egg: copyright hover */}
                <div className="footer__copyright">
                    <span className="footer__default">© 2026</span>
                    <span className="footer__hover">
                        Every great answer begins with a great question.
                    </span>
                </div>
            </div>
        </footer>
    );
}

export default Footer;
