import { useEffect, useState } from "react";
// Public Dataset APIs used by the Home page.
import { getDataset, getPublicDocument } from "../services/auth";


function Dataset()
{

// Stores the list of documents returned by the backend
const [documents, setDocuments] = useState([]);

// Stores the currently selected document
const [selectedDocument, setSelectedDocument] = useState(null);

// Fetch the complete public dataset when the page loads
useEffect(() => {

    async function loadDataset() {

        try {

            const data = await getDataset();

            setDocuments(data.documents);

        } catch (error) {

            console.error("Failed to load dataset:", error);

        }

    }

    loadDataset();

}, []);


// Fetches and displays the contents of the selected document.
// The Dataset page initially loads only document metadata.
// When a user clicks a document, the complete document is
// retrieved from the backend.
async function openDocument(document_id) {

    // Close the document if it is already open.
    if (selectedDocument?.id === document_id) {

        setSelectedDocument(null);

        return;

    }

    try {

        // Request the full document from the public Dataset endpoint.
        const data = await getPublicDocument(document_id);

        // Store the selected document so its contents can be displayed.
        setSelectedDocument(data);

    } catch (error) {

        console.error("Failed to load document:", error);

    }

}



return (
<main>

    <h1>Dataset</h1>

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






export default Dataset;