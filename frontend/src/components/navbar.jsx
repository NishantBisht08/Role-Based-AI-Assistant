import { NavLink, useNavigate } from "react-router-dom";   //we can move to different pages using navlink
//whole page doesn't need to refresh

import { useAuth } from "../contexts/AuthContext";

import { logout } from "../services/auth";


function Navbar() {

    const navigate = useNavigate();

    const { user, setUser } = useAuth();


    const handleLogout = async () => {

        const confirmed = window.confirm(
            "Are you sure you want to log out?"
        ); 


    if (!confirmed) {
    return;
    }
 
    try {

        await logout();

        setUser(null);

        navigate("/");

    }

    catch (err) {

        console.error(err);

    }

};


    return (
        <header>

            <h1>Role-Based AI Assistant</h1>

            <nav>

                <NavLink to="/">
                    Home
                </NavLink>

                
                  {user && (

                  <NavLink to="/dashboard">
                          Dashboard
                  </NavLink>
                  )}
                

                <NavLink to="/about">
                    About
                </NavLink>


            </nav>

            <div>

        {
           user ? (

               <>
                    <span>
                         {user.name}
                    </span>

                    <button onClick={handleLogout}>
                        Logout
                    </button>
               </>

               ) : (

                  <NavLink to="/login">
                        Login
                  </NavLink>

                )
        }

           </div>

        </header>
    );
}

export default Navbar;