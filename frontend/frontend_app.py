import streamlit as st
import requests
import json
import base64
from datetime import datetime, timezone

# ── Configuration ──
import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
if not BACKEND_URL.startswith("http"):
    BACKEND_URL = f"https://{BACKEND_URL}"

st.set_page_config(
    page_title="FinSolve Enterprise AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for FinTech Look ──
st.markdown("""
<style>
    /* Dark slate and teal theme */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .stSidebar {
        background-color: #1e293b !important;
    }
    .stButton>button {
        background-color: #0d9488;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0f766e;
        transform: translateY(-1px);
    }
    .title-banner {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #0d9488;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .role-finance { background-color: #064e3b; color: #34d399; }
    .role-hr { background-color: #701a75; color: #f0abfc; }
    .role-c-level { background-color: #78350f; color: #fbbf24; }
    .role-marketing { background-color: #1e3a8a; color: #93c5fd; }
    .role-engineering { background-color: #3f6212; color: #bef264; }
    .role-employee { background-color: #334155; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ──
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "refresh_token" not in st.session_state:
    st.session_state["refresh_token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Helper Functions ──
def login(username, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/login", 
            json={"emp_id": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
            st.session_state["username"] = username
            st.session_state["messages"] = []
            return True, "Login successful"
        else:
            return False, response.json().get("detail", "Login failed")
    except Exception as e:
        return False, f"Connection error: Make sure the FastAPI backend is running. ({e})"

def logout():
    st.session_state["access_token"] = None
    st.session_state["refresh_token"] = None
    st.session_state["username"] = None
    st.session_state["messages"] = []
    st.rerun()

def refresh_session():
    if not st.session_state["refresh_token"]:
        return False
    try:
        response = requests.post(
            f"{BACKEND_URL}/refresh",
            json={"refresh_token": st.session_state["refresh_token"]}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state["access_token"] = data["access_token"]
            st.session_state["refresh_token"] = data["refresh_token"]
            return True
        else:
            return False
    except:
        return False

def ask_question(query):
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={"token": st.session_state["access_token"], "question": query}
        )
        
        # Token expired -> try to refresh automatically
        if response.status_code == 401:
            if refresh_session():
                # Retry with new token
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"token": st.session_state["access_token"], "question": query}
                )
            else:
                st.error("Session expired entirely (absolute timeout or invalid refresh token). Please log in again.")
                logout()
                return None
                
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            st.error("Account locked or access denied.")
            return None
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
            
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        return None

# ── Login Screen ──
if not st.session_state["access_token"]:
    st.markdown('<div class="title-banner"><h1>🔒 FinSolve Secure Portal</h1><p>Enterprise AI Assistant Authentication</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            st.subheader("Sign In")
            username = st.text_input("Employee ID")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate")
            
            if submit:
                if username and password:
                    with st.spinner("Verifying credentials..."):
                        success, msg = login(username, password)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter both username and password.")
                    
# ── Main Chat Interface ──
else:
    def get_role_from_token(token):
        try:
            payload = token.split(".")[1]
            payload += "=" * ((4 - len(payload) % 4) % 4)
            decoded = json.loads(base64.b64decode(payload).decode('utf-8'))
            return decoded.get("role", "employee")
        except:
            return "employee"
            
    current_role = get_role_from_token(st.session_state["access_token"])
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state['username'].title()}")
        st.markdown(f'<span class="role-badge role-{current_role}">{current_role} Clearance</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("#### Accessible Departments")
        role_map = {
            "finance": ["Finance"],
            "hr": ["HR", "General"],
            "engineering": ["Engineering"],
            "marketing": ["Marketing"],
            "employee": ["General"],
            "c-level": ["Finance", "HR", "Engineering", "Marketing", "General"]
        }
        allowed = role_map.get(current_role, ["General"])
        for dept in allowed:
            st.markdown(f"- 📁 {dept}")
            
        st.markdown("---")
        if st.button("🚪 Secure Logout", use_container_width=True):
            logout()
            
    # Main content
    st.markdown('<div class="title-banner"><h1>💼 FinSolve AI Assistant</h1><p>Ask questions about your authorized departmental data.</p></div>', unsafe_allow_html=True)

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Cited Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- 📄 `{src}`")

    if prompt := st.chat_input("Ask a question about your departmental data..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Searching secure database..."):
                response_data = ask_question(prompt)
                
            if response_data:
                answer = response_data.get("answer", "No answer generated.")
                sources = response_data.get("sources", [])
                
                message_placeholder.markdown(answer)
                if sources:
                    with st.expander("View Cited Sources"):
                        for src in sources:
                            st.markdown(f"- 📄 `{src}`")
                            
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
