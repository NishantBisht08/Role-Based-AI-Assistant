import api from "./api";

// Login user
export async function login(emp_id, password) {

    const response = await api.post("/login", {
        emp_id,
        password,
    });

    return response.data;
}

// Fetch the currently authenticated user
export async function getCurrentUser() {

    const response = await api.get("/me");

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
