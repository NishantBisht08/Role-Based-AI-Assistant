import os  # Used for working with folders and file paths

# Reuse existing helper to extract metadata (quarter, year, document type)
from backend.rag_engine.document_loader import extract_file_tags

# Import RBAC mapping to filter documents based on user role
from backend.rag_engine.rbac import ROLE_FOLDERS


# Path to the project's data/ directory
BASE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data"
)

# File types supported by the Dataset page
SUPPORTED_EXTENSIONS = (
    ".md",
    ".txt",
    ".csv",
    ".pdf",
    ".docx",
)



# ── List All Documents ────────────────────────────────────────────────────────
# This function scans the entire data/ directory and collects metadata
# for every supported document.
#
# It does NOT perform any RBAC checks.
# It is used by the public Dataset page on the Home screen, where all
# documents are visible.
#
# The function returns a list of dictionaries containing document
# metadata (title, folder, filename, quarter, year, etc.).

def list_all_documents():

    # Stores metadata for every discovered document
    documents = []

    # Iterate through every folder inside the data directory
    # Example: engineering/, finance/, hr/, general/, marketing/
    for folder in sorted(os.listdir(BASE_DATA_PATH)):

        folder_path = os.path.join(BASE_DATA_PATH, folder)

        # Ignore anything that isn't a folder
        if not os.path.isdir(folder_path):
            continue

        # Iterate through every file inside the current folder
        for filename in sorted(os.listdir(folder_path)):

            # Build the complete path to the current file
            file_path = os.path.join(folder_path, filename)

            # Skip anything that isn't a file
            if not os.path.isfile(file_path):
                continue

            # Skip unsupported file types
            if not filename.endswith(SUPPORTED_EXTENSIONS):
                continue

            # Extract metadata (quarter, year, document type)
            # from the filename using the existing RAG helper
            tags = extract_file_tags(filename)

            # Convert filename into a cleaner title for display
            # Example:
            # engineering_master_doc.md
            # → Engineering Master Doc
            title = (
                filename
                .replace(".md", "")
                .replace(".csv", "")
                .replace(".pdf", "")
                .replace(".docx", "")
                .replace(".txt", "")
                .replace("_", " ")
                .title()
            )

            # Create a stable document ID
            # Example:
            # engineering_engineering_master_doc
            document_id = f"{folder}_{os.path.splitext(filename)[0]}"

            # Store all useful metadata about this document
            document = {
                "id": document_id,
                "title": title,
                "filename": filename,
                "folder": folder,
                "quarter": tags["quarter"],
                "year": tags["year"],
                "doc_type": tags["doc_type"],
            }

            # Add the document metadata to the final list
            documents.append(document)

    # Return metadata for every document found
    return documents




def list_role_documents(role):
    
    # Convert role to lowercase and remove any extra spaces
    role = role.strip().lower()

    # Get the folders this role is allowed to access
    # If the role doesn't exist, return an empty list
    allowed_folders = ROLE_FOLDERS.get(role, [])

    # Get metadata for every document in the data folder
    documents = list_all_documents()

    # Store only the documents this role is allowed to view
    filtered_documents = []

    # Check every document
    for document in documents:

        # If the document's folder is accessible to this role
        if document["folder"] in allowed_folders:

            # Save it in the filtered list
            filtered_documents.append(document)

    # Return only the accessible documents
    return filtered_documents
    
    
    # ── Get Document Content ──────────────────────────────────────────────────────
# Returns the contents of a single document after verifying
# that the user's role has permission to access it.

def get_document(document_id, role):
    
    # Normalize the role name
    role = role.strip().lower()
    
    # Get folders accessible by this role
    allowed_folders = ROLE_FOLDERS.get(role, [])
    
    
    # Split the document ID into folder and filename, example:  "finance_quarterly_financial_report" :- THe first part here is the folder
    parts = document_id.split("_", 1)
    
    # Invalid document ID
    if len(parts) != 2:
        return None
    
    # Extract folder name and filename from the document ID
    folder = parts[0]
    filename = parts[1]
    
    # Deny access if the role is not allowed to view this folder
    if folder not in allowed_folders:
        return None                       
    
    # Find the actual file by checking supported extensions
    file_path = None
    
    
    for extension in SUPPORTED_EXTENSIONS:
        candidate_path = os.path.join(
            BASE_DATA_PATH,
            folder,
            filename + extension
        )
        
        if os.path.isfile(candidate_path):
            file_path = candidate_path
            break
        
    
    # Return None if the document doesn't exist
    if file_path is None:
        return None
    
    # Read the document contents
    with open(file_path, "r", encoding="utf-8") as file:
         content = file.read()
         
         
    # Find the document metadata
    documents = list_all_documents()

    for document in documents:

        if document["id"] == document_id:
            document["content"] = content
            return document
        
    return None



# Returns a single document from the public dataset.
# Unlike get_document(), this function does not perform
# any authentication or RBAC checks because it is used
# by the public Dataset page on the Home screen.
def get_public_document(document_id):
    
    # Split the document ID into folder and filename.
    parts = document_id.split("_", 1)

    if len(parts) != 2:
        return None

    folder = parts[0]
    filename = parts[1]

    # Stores the file path if found.
    file_path = None

    # Search for the document using the supported extensions.
    for extension in SUPPORTED_EXTENSIONS:

        candidate_path = os.path.join(
            BASE_DATA_PATH,
            folder,
            filename + extension
        )

        if os.path.isfile(candidate_path):

            file_path = candidate_path

            break

    # Document was not found.
    if file_path is None:
        return None

    # Read the document contents.
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Retrieve document metadata.
    documents = list_all_documents()

    for document in documents:

        if document["id"] == document_id:

            document["content"] = content

            return document

    return None