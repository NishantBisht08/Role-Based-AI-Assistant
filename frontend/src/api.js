import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

// ✅ ADD ACCESS TOKEN TO EVERY REQUEST
API.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");

  if (token) {
    if (!config.data) config.data = {};
    config.data.token = token;
  }

  return config;
});

// ✅ AUTO REFRESH TOKEN
API.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response && error.response.status === 401) {
      const refresh_token = localStorage.getItem("refresh_token");

      try {
        const res = await axios.post("http://127.0.0.1:8000/refresh", {
          refresh_token: refresh_token,
        });

        // SAVE NEW TOKENS
        localStorage.setItem("access_token", res.data.access_token);
        localStorage.setItem("refresh_token", res.data.refresh_token);

        // RETRY ORIGINAL REQUEST
        if (!error.config.data) error.config.data = {};
        error.config.data.token = res.data.access_token;

        return axios(error.config);

      } catch (err) {
        // logout if refresh fails
        localStorage.clear();
        window.location.href = "/";
      }
    }

    return Promise.reject(error);
  }
);

export default API;