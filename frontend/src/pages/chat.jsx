import { useState } from "react";

import { useAuth } from "../contexts/AuthContext";



function Chat() {

const { user } = useAuth();

const [messages, setMessages] = useState([]);

const [input, setInput] = useState("");

const [loading, setLoading] = useState(false);

const [error, setError] = useState("");

const [showDocuments, setShowDocuments] = useState(false);


const handleSend = () => {

    if (!input.trim()) {
        return;
    }

    const userMessage = {

        role: "user",

        content: input,

    };

    setMessages((previousMessages) => [

        ...previousMessages,

        userMessage,

    ]);

    setInput("");

};



return (

<main>

    <section>

        {/*Header*/}
          
               <h2>
                    Role-Based AI Assistant
                </h2>

          <button>
              Documents
          </button> 

    </section>


{messages.length === 0 && (

    <section>

         <h2>
        Ready when you are, {user.name}.
    </h2>

    <p>
        Ask a question about the documents you have access to.
    </p>

    </section>

)}

<section>

     {messages.map((message, index) => (

        <div key={index}>

            <strong>

                {message.role === "user" ? "You" : "Assistant"}

            </strong>

            <p>

                {message.content}

            </p>

        </div>

    ))}
 
</section>


    <section>

        <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask a question about your company documents..." >



        </textarea>


        <button  onClick={handleSend}  disabled={!input.trim() || loading}>

              Send

        </button>


    </section>

</main>

);


}

export default Chat;