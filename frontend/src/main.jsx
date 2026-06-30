import { StrictMode } from 'react'  //it checks for unsafe code, bugs, it doesn't exist in production only while we are writing code
import { createRoot } from 'react-dom/client'   //gives react the html element where the application should live
import './index.css'  //loads the global stylesheet, used by every page in our application
import App from './App.jsx'  //imports our entire application
import { AuthProvider } from "./contexts/AuthContext";

//this line finds the div which is present inside index.html
createRoot(document.getElementById('root')).render(  
  <StrictMode>
      <AuthProvider>
            <App />
      </AuthProvider>

  </StrictMode>
)
//react then renders app inside that div



