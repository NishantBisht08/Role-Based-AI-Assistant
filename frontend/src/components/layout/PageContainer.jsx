/**
 * PageContainer — Standard page wrapper with consistent max-width and padding
 *
 * Every page uses this to maintain uniform horizontal spacing and
 * responsive content width across the application.
 */
function PageContainer({ children, narrow = false, wide = false, className = "" }) {
    const classes = [
        "page-container",
        narrow ? "page-container--narrow" : "",
        wide ? "page-container--wide" : "",
        className,
    ]
        .filter(Boolean)
        .join(" ");

    return <main className={classes}>{children}</main>;
}

export default PageContainer;
