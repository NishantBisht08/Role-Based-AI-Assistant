import { useEffect, useState } from "react";

// Import the API functions used by this page
// getRoleDocuments -> Returns only the documents the logged-in user is allowed to see
// getDocument -> Returns the full contents of one selected document
import { getRoleDocuments, getDocument } from "../services/auth";


function DashboardDataset()
{

    // Stores the list of documents returned by the backend.
// These are only the documents the user's role is allowed to access.
const [documents, setDocuments] = useState([]);

// Stores the currently opened document.
// Initially no document is selected.
const [selectedDocument, setSelectedDocument] = useState(null);


// Fetch the documents that the logged-in user's role
// is allowed to access when the page first loads.
useEffect(() => {

    async function loadDocuments() {

        try {

            // Ask the backend for only the accessible documents.
            const data = await getRoleDocuments();

            // Save the returned document metadata in React state.
            setDocuments(data.documents);

        } catch (error) {

            console.error("Failed to load documents:", error);

        }

    }

    loadDocuments();

}, []);

// Fetch the full contents of the document
// that the user clicked.
async function openDocument(document_id) {

    // If the user clicks the currently opened document,
    // close it instead of fetching it again.
    if (
        selectedDocument &&
        selectedDocument.id === document_id
    ) {

        setSelectedDocument(null);  //if user clicks document for first time, value is null, so documents are fetched, for 2nd time, currently opened doc is closed

        return;

    }

    try {

        // Ask the backend for one specific document.
        const data = await getDocument(document_id);

        // Save the document in state so React displays it.
        setSelectedDocument(data);

    } catch (error) {

        console.error("Failed to load document:", error);

    }

}


return (
<main>

    {/* Displays only the documents the logged-in user
    is allowed to access */}
<h1>Accessible Documents</h1>

{
    documents.map((document) => (
        <div key={document.id}>

<button
    type="button"
    onClick={() => openDocument(document.id)}
>
    {document.title}

</button>

        </div>

        ))
}



{
    selectedDocument && (

        <section>

            <h2>{selectedDocument.title}</h2>

            <pre>

                {selectedDocument.content}

            </pre>

        </section>

    )
}



</main>

);


}

export default DashboardDataset;