/**
 * Card — Reusable content container with optional hover elevation
 *
 * Used for feature highlights, dashboard sections, and info blocks.
 * The card grid on pages limits to 3 per row via CSS grid.
 */
function Card({
    children,
    icon,
    title,
    description,
    hover = true,
    glass = false,
    className = "",
    onClick,
}) {
    const classes = [
        "card",
        hover ? "card--hover" : "",
        glass ? "card--glass" : "",
        className,
    ]
        .filter(Boolean)
        .join(" ");

    return (
        <div className={classes} onClick={onClick} role={onClick ? "button" : undefined}>
            {icon && <div className="card__icon">{icon}</div>}
            {title && <h3 className="card__title">{title}</h3>}
            {description && <p className="card__description">{description}</p>}
            {children}
        </div>
    );
}

export default Card;
