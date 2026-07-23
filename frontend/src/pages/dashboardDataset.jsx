/**
 * DashboardDataset — RBAC-protected dataset viewer
 *
 * Shows only documents the authenticated user's role can access.
 * Uses getRoleDocuments() and getDocument() — RBAC enforced by backend.
 */
import { useEffect, useState } from "react";
import { getRoleDocuments, getDocument } from "../services/auth";
import PageContainer from "../components/layout/PageContainer";
import ParchmentPaper from "../components/common/ParchmentPaper";
import DocumentViewer from "../components/common/DocumentViewer";

function DashboardDataset() {
    const [documents, setDocuments] = useState([]);
    const [selectedDocument, setSelectedDocument] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function loadDocuments() {
            try {
                const data = await getRoleDocuments();
                setDocuments(data.documents);
            } catch (error) {
                console.error("Failed to load documents:", error);
            } finally {
                setLoading(false);
            }
        }
        loadDocuments();
    }, []);

    async function openDocument(document_id) {
        if (selectedDocument && selectedDocument.id === document_id) {
            setSelectedDocument(null);
            return;
        }

        try {
            const data = await getDocument(document_id);
            setSelectedDocument(data);
        } catch (error) {
            console.error("Failed to load document:", error);
        }
    }

    return (
        <PageContainer wide>
            <div className="section-header">
                <h2>Accessible Documents</h2>
                <p>Documents available to your role. Access is enforced by RBAC.</p>
            </div>

            {loading ? (
                <div className="card-grid">
                    {[1, 2, 3].map((index) => (
                        <div key={index} className="skeleton" style={{ height: "140px" }} />
                    ))}
                </div>
            ) : documents.length === 0 ? (
                <div className="card text-center" style={{ padding: "3rem" }}>
                    <p className="text-muted">
                        No documents accessible for your current role.
                    </p>
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

export default DashboardDataset;
