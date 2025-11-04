"""
Document Ingestion Script
Processes PDF, HTML, and code files and creates a vector database for RAG
"""
import os
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    """Load documents from the data directory (PDF, HTML, and code files)"""
    print("Loading documents...")
    documents = []
    
    if not os.path.exists(DATA_PATH):
        print(f"Creating {DATA_PATH} directory...")
        os.makedirs(DATA_PATH)
        return documents
    
    files = os.listdir(DATA_PATH)
    if not files:
        print("No files found in data directory")
        return documents
    
    # Load PDFs
    pdf_files = [f for f in files if f.endswith('.pdf')]
    if pdf_files:
        print(f"Loading {len(pdf_files)} PDF file(s)...")
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
    
    # Load HTML files
    html_files = [f for f in files if f.endswith(('.html', '.htm'))]
    if html_files:
        print(f"Loading {len(html_files)} HTML file(s)...")
        for html_file in html_files:
            try:
                loader = UnstructuredHTMLLoader(os.path.join(DATA_PATH, html_file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {html_file}: {e}")
    
    # Load code and text files
    code_extensions = ['.py', '.java', '.js', '.cpp', '.c', '.h', '.cs', '.rb', 
                      '.go', '.rs', '.php', '.swift', '.kt', '.ts', '.txt', '.md']
    code_files = [f for f in files if any(f.endswith(ext) for ext in code_extensions)]
    if code_files:
        print(f"Loading {len(code_files)} code/text file(s)...")
        for code_file in code_files:
            try:
                loader = TextLoader(os.path.join(DATA_PATH, code_file), encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {code_file}: {e}")
    
    print(f"Loaded {len(documents)} document sections total")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks"""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create and persist vector store"""
    print("Creating embeddings and vector store...")
    
    # Delete existing vector store if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing vector store at {CHROMA_PATH}...")
        import shutil
        shutil.rmtree(CHROMA_PATH)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Changed to CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Vector store created and saved to {CHROMA_PATH}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=" * 50)
    print("Starting Document Ingestion Pipeline")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} directory not found!")
        return
    
    # Load, split, and store documents
    documents = load_documents()
    if not documents:
        print("No documents found to process!")
        return
    
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)
    
    print("=" * 50)
    print("Document ingestion completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
