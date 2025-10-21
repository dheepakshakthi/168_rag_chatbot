"""
Document Ingestion Script
Processes PDF files and creates a vector database for RAG
"""
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    """Load PDF documents from the data directory"""
    print("Loading PDF documents...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")
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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
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
