import api from "./api";

// Login user
export async function login(emp_id, password) {
    const response = await api.post("/login", {
        emp_id,
        password,
    });

    return response.data;
}

// Fetch the currently authenticated user.
//
// Refresh is still attempted if the access token expired.
// If the visitor has no valid session, the interceptor will not
// force a public page to redirect to /login.
export async function getCurrentUser() {
    const response = await api.get("/me", {
        suppressAuthRedirect: true,
    });

    return response.data;
}

export async function logout() {
    const response = await api.post("/logout");

    return response.data;
}

export async function refresh() {
    const response = await api.post("/refresh");

    return response.data;
}


export async function askQuestion(question) {

    const response = await api.post("/ask", {

        question,

    });

     return response.data

}



export async function setPassword(emp_id, new_password) {
    const response = await api.post("/set-password", {
        emp_id,
        new_password,
    });

    return response.data;
}


export async function changePassword(emp_id, old_password, new_password) {
    const response = await api.post("/change-password", {
        emp_id,
        old_password,
        new_password,
    });

    return response.data;
}

