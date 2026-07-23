/**
 * Button — Primary interactive element across Novaris
 *
 * Supports multiple visual variants and sizes. Includes a subtle
 * ripple effect on click for premium micro-interaction feedback.
 */
function Button({
    children,
    variant = "primary",
    size = "md",
    className = "",
    onClick,
    disabled = false,
    type = "button",
    ...props
}) {
    const classes = [
        "btn",
        `btn--${variant}`,
        size === "lg" ? "btn--lg" : size === "sm" ? "btn--sm" : "",
        className,
    ]
        .filter(Boolean)
        .join(" ");

    function handleClick(event) {
        // Ripple micro-interaction
        const button = event.currentTarget;
        const ripple = document.createElement("span");
        ripple.classList.add("ripple");
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        ripple.style.width = ripple.style.height = `${size}px`;
        ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
        ripple.style.top = `${event.clientY - rect.top - size / 2}px`;
        button.appendChild(ripple);
        ripple.addEventListener("animationend", () => ripple.remove());

        onClick?.(event);
    }

    return (
        <button
            type={type}
            className={classes}
            onClick={handleClick}
            disabled={disabled}
            {...props}
        >
            {children}
        </button>
    );
}

export default Button;
