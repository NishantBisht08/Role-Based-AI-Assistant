/**
 * Dataset — Public dataset viewer (Home page access)
 *
 * Displays all documents as parchment papers (2-3 per row).
 * Uses getDataset() and getPublicDocument() — no auth required.
 */
import { useEffect, useState } from "react";
import { getDataset, getPublicDocument } from "../services/auth";
import PageContainer from "../components/layout/PageContainer";
import ParchmentPaper from "../components/common/ParchmentPaper";
import DocumentViewer from "../components/common/DocumentViewer";

function Dataset() {
    const [documents, setDocuments] = useState([]);
    const [selectedDocument, setSelectedDocument] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function loadDataset() {
            try {
                const data = await getDataset();
                setDocuments(data.documents);
            } catch (error) {
                console.error("Failed to load dataset:", error);
            } finally {
                setLoading(false);
            }
        }
        loadDataset();
    }, []);

    async function openDocument(document_id) {
        if (selectedDocument?.id === document_id) {
            setSelectedDocument(null);
            return;
        }

        try {
            const data = await getPublicDocument(document_id);
            setSelectedDocument(data);
        } catch (error) {
            console.error("Failed to load document:", error);
        }
    }

    return (
        <PageContainer wide>
            <div className="section-header">
                <h2>Dataset</h2>
                <p>This is the complete dataset used in Novaris.</p>
            </div>

            {loading ? (
                <div className="card-grid">
                    {[1, 2, 3, 4, 5, 6].map((index) => (
                        <div key={index} className="skeleton" style={{ height: "140px" }} />
                    ))}
                </div>
            ) : (
                <div className="parchment-grid">
                    {documents.map((document) => (
                        <ParchmentPaper
                            key={document.id}
                            title={document.title}
                            onClick={() => openDocument(document.id)}
                        />
                    ))}
                </div>
            )}

            {selectedDocument && (
                <DocumentViewer
                    title={selectedDocument.title}
                    content={selectedDocument.content}
                    onClose={() => setSelectedDocument(null)}
                />
            )}
        </PageContainer>
    );
}

export default Dataset;
