import axios from "axios";

const baseURL = import.meta.env.VITE_API_URL;

const api = axios.create({
    baseURL,
    withCredentials: true,
    headers: {
        "Content-Type": "application/json",
    },
});

// Separate client without interceptors.
// This prevents /refresh from triggering its own interceptor.
const refreshClient = axios.create({
    baseURL,
    withCredentials: true,
    headers: {
        "Content-Type": "application/json",
    },
});

// Concurrent failed requests share the same refresh operation.
let refreshPromise = null;

const excludedEndpoints = [
    "/login",
    "/logout",
    "/refresh",
    "/set-password",
    "/change-password",
];

function getRequestPath(config) {
    try {
        return new URL(config.url, config.baseURL || baseURL).pathname;
    } catch {
        return config.url;
    }
}

api.interceptors.response.use(
    (response) => response,

    async (error) => {
        const originalRequest = error.config;

        if (!originalRequest) {
            return Promise.reject(error);
        }

        const requestPath = getRequestPath(originalRequest);
        const isExcluded = excludedEndpoints.includes(requestPath);

        if (
            error.response?.status !== 401 ||
            originalRequest._retry ||
            isExcluded
        ) {
            return Promise.reject(error);
        }

        originalRequest._retry = true;

        try {
            if (!refreshPromise) {
                refreshPromise = refreshClient
                    .post("/refresh")
                    .finally(() => {
                        refreshPromise = null;
                    });
            }

            await refreshPromise;

            return api(originalRequest);
        } catch (refreshError) {
            /*
             * AuthProvider uses suppressAuthRedirect for its startup /me check.
             * An unauthenticated visitor on a public page should not be
             * forcefully redirected to /login.
             */
            if (
                !originalRequest.suppressAuthRedirect &&
                window.location.pathname !== "/login"
            ) {
                window.location.replace("/login");
            }

            return Promise.reject(refreshError);
        }
    }
);

export default api;