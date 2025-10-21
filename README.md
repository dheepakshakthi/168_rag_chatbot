# RAG Chatbot for Coding Questions

A Retrieval-Augmented Generation (RAG) chatbot that answers coding questions based on your PDF learning materials using the SmolLM2:1.7b model from Ollama.

## Features

- 📚 **PDF Processing**: Automatically processes PDF documents from the `data/` folder
- 🔍 **Semantic Search**: Uses vector embeddings to find relevant content
- 🤖 **Local LLM**: Powered by SmolLM2:1.7b via Ollama (runs locally)
- 💬 **Interactive Chat**: Simple command-line interface for asking questions
- 🎯 **Context-Aware**: Answers based on your specific learning materials

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed with SmolLM2:1.7b model
   - Install Ollama from: https://ollama.ai
   - Pull the model: `ollama pull smollm2:1.7b`

## Setup

### 1. Install Dependencies

```powershell
# Activate virtual environment
.\master\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your PDF files in the `data/` folder. Currently includes:
- `learning_java.pdf`
- `Learning_Python.pdf`

### 3. Ingest Documents

Process the PDFs and create the vector database:

```powershell
python ingest_documents.py
```

This will:
- Load all PDF files from the `data/` folder
- Split them into manageable chunks
- Create embeddings using sentence-transformers
- Store them in a ChromaDB vector database

## Usage

### Start the Chatbot

```powershell
python chatbot.py
```

### Example Questions

- "How do I create a list in Python?"
- "What is inheritance in Java?"
- "Explain Python decorators"
- "How do I handle exceptions in Java?"
- "What are Python list comprehensions?"
- "Explain Java interfaces"

### Commands

- Type your question and press Enter
- Type `sources` to see the source documents for the last answer
- Type `quit` or `exit` to end the conversation

## How It Works

1. **Document Ingestion** (`ingest_documents.py`):
   - Loads PDF files
   - Splits into chunks (1000 chars with 200 overlap)
   - Creates embeddings using all-MiniLM-L6-v2
   - Stores in ChromaDB

2. **RAG Pipeline** (`chatbot.py`):
   - Takes user question
   - Finds relevant chunks using semantic search
   - Creates context from top 4 results
   - Sends context + question to SmolLM2:1.7b
   - Returns contextual answer

## Project Structure

```
rag_chatbot/
├── data/                      # PDF documents
│   ├── learning_java.pdf
│   └── Learning_Python.pdf
├── chroma_db/                 # Vector database (created after ingestion)
├── master/                    # Virtual environment
├── ingest_documents.py        # Document processing script
├── chatbot.py                # Main chatbot application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Troubleshooting

### "Vector store not found" error
Run `python ingest_documents.py` first to create the vector database.

### Ollama connection error
- Make sure Ollama is running
- Verify the model is installed: `ollama list`
- Pull the model if needed: `ollama pull smollm2:1.7b`

### Out of memory errors
- Reduce `TOP_K_RESULTS` in `chatbot.py` (default: 4)
- Reduce `chunk_size` in `ingest_documents.py` (default: 1000)

## Customization

### Change the LLM Model
Edit `OLLAMA_MODEL` in `chatbot.py`:
```python
OLLAMA_MODEL = "your-model-name"
```

### Adjust Chunk Size
Edit `chunk_size` in `ingest_documents.py`:
```python
chunk_size=1000,  # Increase or decrease
chunk_overlap=200  # Adjust overlap
```

### Change Number of Retrieved Documents
Edit `TOP_K_RESULTS` in `chatbot.py`:
```python
TOP_K_RESULTS = 4  # Increase for more context
```

## Adding More Documents

1. Add PDF, text, or code files to the `data/` folder
2. Re-run the ingestion: `python ingest_documents.py`
3. The chatbot will now include the new materials

## License

MIT License - Feel free to use and modify as needed!
