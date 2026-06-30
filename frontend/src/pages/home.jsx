import { Link } from "react-router-dom";

//features using title cards (feature array)
const features = [
    {
        title: "JWT Authentication",
        description: "Secure login using Access and Refresh Tokens."
    },

    {
        title: "Role-Based Access",
        description: "Users can only access documents permitted by their role."
    },

    {
        title: "AI-Powered RAG",
        description: "Answers are generated only from authorized documents."
    },

    {
        title: "FastAPI Backend",
        description: "High-performance backend with modular architecture."
    },

    {
        title: "PostgreSQL Database",
        description: "Secure storage for users, sessions and roles."
    },

    {
        title: "LangChain + ChromaDB",
        description: "Retrieves relevant documents using vector search."
    }
];


const reasons = [
    {
        title: "Role-Aware Responses",
        description:
            "Answers are generated according to the permissions assigned to your role."
    },

    {
        title: "Contextual Question Answering",
        description:
            "Uses Retrieval-Augmented Generation to answer questions from company documents."
    },

    {
        title: "Secure Session Management",
        description:
            "JWT authentication with access and refresh tokens protects user sessions."
    },

    {
        title: "Enterprise-Ready Architecture",
        description:
            "Designed with modular FastAPI services and scalable database support."
    },

    {
        title: "Fast Information Retrieval",
        description:
            "Relevant documents are retrieved before generating AI responses."
    },

    {
        title: "Administrative Control",
        description:
            "Administrators can create users and manage organizational access."
    },

    
];


const facts = [
    {
        title: "Two users can ask the same question and receive different answers.",
        description: "Responses depend on each user's assigned permissions."
    },

    {
        title: "Vector search finds meaning, not just matching words.",
        description: "Similar ideas can be found even without identical keywords."
    },

    {
        title: "Access tokens have short lifetimes.",
        description: "Short-lived access tokens reduce the risk of unauthorized access."
    },

    {
        title: "The AI can't retrieve what you're not allowed to see.",
        description: "Access control is enforced before retrieval begins."

    },
];



function Home() {
    return (
        <main>

           {/* Hero Section */}

            <section>
 
                <h2>Role-Based AI Assistant</h2>  {/*Heading and tagline for the project*/}

                <h3>Where Security Meets Intelligence.</h3>

                <p>
                    A secure AI assistant powered by Role-Based Access Control (RBAC)
                    and Retrieval-Augmented Generation (RAG).
                </p>

            </section>

            
           {/* Features Section */}
           {/* //going through each feature in array one by one using map, key helps react to give each feature in array a unique identifier
            // React uses this internally to efficiently update the UI when items change. */}
            <section>

                 <h2>Features</h2>

                 {
                       features.map((feature) => (

                       <div key={feature.title}>

                       <h3>{feature.title}</h3>

                       <p>{feature.description}</p> 

                       </div>

                  ))
              }

            </section>


            {/* Demo Guide Section */}
            <section>

                 <h2>Demo Guide</h2>

                <p>
                     New here? Explore the application through a step-by-step walkthrough
                     before logging in.
                </p>

                 <Link to="/guide"> {/* on clicking this , we get directed to demo page */}
                   Enter Demo Guide
                </Link>


            </section>


            {/* Reasons for choosing assistant Section */}
            <section>

                  <h2>Why Choose This Assistant?</h2>

               {
                     reasons.map((reason)=> (

                        <div key={reason.title}>
                             
                             <h3>{reason.title}</h3>

                             <p>{reason.description}</p>   {/* the array is capabilities and each item is capability */}


                        </div>
                    ))
               }

            </section>


            {/* Dataset Section */}
            <section>

                  <h2>Dataset</h2>

                  <p>
                    Explore the documents and datasets that power the AI assistant.
                  </p>

                  <Link to="/dataset">
                      View Dataset
                  </Link>
                    

            </section>


            {/* Do You Know Section */}
            <section>

                       <h2>Did You Know?</h2>

                 {
                      facts.map((fact) => (

                       <div key={fact.title}>

                            <h3>{fact.title}</h3>

                            <p>{fact.description}</p>

                       </div>

                    ))
                 }

            </section>


          {/* Author Section */}
           <section>

                 <h2>Meet the Developers</h2>

                  <p>
                    Learn more about the developer behind the Role-Based AI Assistant project.
                  </p>

                  <Link to="/about-author">
                           About the Author
                  </Link>

           </section>




        </main>
    );
}



export default Home;  //makes this component available to all 