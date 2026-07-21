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

export async function createUser(emp_id, name, role) {

    const response = await api.post(
        "/admin/create-user",
        {
            emp_id,
            name,
            role,
        }
    );

    return response.data;

}


// Fetch the complete public dataset
export async function getDataset() {

    const response = await api.get("/dataset");

    return response.data;

}


// Fetch documents accessible to the authenticated user
export async function getRoleDocuments() {

    const response = await api.get("/documents");

    return response.data;

}


// Fetch the contents of a single document
export async function getDocument(document_id) {

    const response = await api.get(`/documents/${document_id}`);

    return response.data;

}


// Retrieves a single document from the public Dataset page.
// Unlike getDocument(), this endpoint does not require
// authentication or RBAC because it is used by the Home page.
export async function getPublicDocument(document_id) {
    const response = await api.get(`/dataset/${document_id}`);
    return response.data;
}

