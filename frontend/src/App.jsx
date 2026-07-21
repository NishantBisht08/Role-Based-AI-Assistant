// React Router
import { BrowserRouter, Routes, Route } from "react-router-dom";

import ProtectedRoute from "./components/protected_route";

// Components
import Navbar from "./components/navbar";
import Footer from "./components/footer";

// Pages
import Home from "./pages/home";
import About from "./pages/about";
import AboutAuthor from "./pages/about_author";
import DemoGuide from "./pages/demo_guide";
import Login from "./pages/login";
import Dashboard from "./pages/dashboard";
import Chat from "./pages/chat";
import SetPassword from "./pages/set_password";
import ChangePassword from "./pages/change_password";
import CreateUser from "./pages/create_user";
// Dashboard dataset page (RBAC protected)
import DashboardDataset from "./pages/dashboardDataset";
import Dataset from "./pages/dataset";


// Defining all the routes 
function App() {
    return (
        <BrowserRouter>

            <Navbar />

            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/home" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route path="/about-author" element={<AboutAuthor />} />
                <Route path="/demo" element={<DemoGuide />} />
                <Route path="/dataset" element={<Dataset />} />
                <Route path="/login" element={<Login />} />     
               <Route
                   path="/dashboard/dataset"
                   element={
                      <ProtectedRoute>
                      <DashboardDataset />
                      </ProtectedRoute>
                    }
                />  
                    
                <Route
                        path="/dashboard"
                        element={
                            <ProtectedRoute>
                            <Dashboard />
                            </ProtectedRoute>
                         }
                      />

                <Route
                        path="/chat"
                        element={
                            <ProtectedRoute>
                            <Chat/>
                            </ProtectedRoute>
                         }
                      />

                <Route path="/set-password" element={<SetPassword />} />

                <Route
                       path="/change-password"
                       element={
                   <ProtectedRoute>
                   <ChangePassword />
                   </ProtectedRoute>
                }
            />

            <Route 
                    path="/create-user" 
                    element={
                    <ProtectedRoute>
                    <CreateUser />
                    </ProtectedRoute>
                 }
              />

            </Routes>

            <Footer />

        </BrowserRouter>
    );
}

export default App;  //it allows other modules to use the app component




