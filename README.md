# # 168 - General Purpose RAG Chatbot

A beautiful, intelligent AI chatbot with Retrieval-Augmented Generation (RAG) capabilities. Upload your documents (PDFs, HTML, code files) and chat with an AI that understands your content, or use it as a general-purpose chatbot without any documents.

## 🌟 Features

- **Dual Mode Operation**
  - 🟢 **RAG Mode**: Upload documents and get AI responses based on your content
  - 🔵 **General Mode**: Chat with AI without any documents loaded

- **Multiple File Format Support**
  - 📄 PDF Documents
  - 🌐 HTML Files (.html, .htm)
  - 💻 Code Files (.py, .java, .js, .cpp, .c, .h, .cs, .rb, .go, .rs, .php, .swift, .kt, .ts)
  - 📝 Text & Markdown (.txt, .md)

- **Beautiful Web Interface**
  - Modern, responsive design
  - File upload with drag-and-drop support
  - Real-time chat with typing indicators
  - Source citations for RAG responses

- **Powerful Technology Stack**
  - 🤖 Ollama (SmolLM2) for AI responses
  - 🔍 ChromaDB for vector storage
  - 📊 Sentence Transformers for embeddings
  - 🌐 Flask for web interface

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running ([Download Ollama](https://ollama.ai/))
3. Pull the SmolLM2 model:
   ```bash
   ollama pull smollm2:135m
   # or for better quality (CLI only):
   ollama pull smollm2:1.7b
   ```

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5000`

## 📖 Usage

### Web Interface

1. **Upload Documents** (Optional)
   - Click the "Upload Files" button
   - Select one or more files (PDF, HTML, code files, etc.)
   - Wait for processing (creates vector embeddings)
   - RAG mode is now active! 🟢

2. **Chat with the AI**
   - Type your question in the input box
   - Press Enter to send
   - Get responses based on your documents (RAG mode) or general AI knowledge

3. **Manage Documents**
   - **Clear Docs**: Remove all uploaded documents and return to general mode
   - **Clear Chat**: Clear the conversation history

### Command Line Interface

Run the chatbot in terminal mode:

```bash
python chatbot.py
```

Commands:
- Type your questions naturally
- `sources` - View sources used in the last RAG response
- `quit` or `exit` - Exit the chatbot

### Manual Document Ingestion

To manually process documents from the `data/` folder:

1. Place your files in the `data/` directory
2. Run:
   ```bash
   python ingest_documents.py
   ```

## 📁 Project Structure

```
168_rag_chatbot/
├── app.py                  # Flask web application
├── chatbot.py             # CLI chatbot interface
├── ingest_documents.py    # Document processing script
├── requirements.txt       # Python dependencies
├── data/                  # Upload folder for documents
├── chroma_db/            # Vector database storage
├── templates/
│   ├── index.html        # Main chat interface
│   └── error.html        # Error page
└── static/
    ├── style.css         # Styling
    └── script.js         # Frontend logic
```

## 🔧 Configuration

Edit the configuration variables in `app.py`, `chatbot.py`, or `ingest_documents.py`:

```python
# Model Configuration
OLLAMA_MODEL = "smollm2:135m"          # Ollama model to use
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embeddings model
TOP_K_RESULTS = 4                      # Number of context chunks to retrieve

# Paths
DATA_PATH = "data/"                    # Document upload folder
CHROMA_PATH = "chroma_db"             # Vector database location
```

## 🎯 Use Cases

- **Learning Assistant**: Upload course materials and get answers based on your lessons
- **Code Documentation**: Upload code files and ask questions about implementation
- **Research Helper**: Process research papers and get insights
- **Knowledge Base**: Create a searchable knowledge base from your documents
- **General Chat**: Use without documents for general AI assistance

## 🔒 Privacy

- All processing happens locally on your machine
- Documents are stored locally in the `data/` folder
- No data is sent to external services (except Ollama API calls)

## 🛠️ Troubleshooting

### "Ollama connection error"
- Make sure Ollama is running: `ollama serve`
- Verify the model is downloaded: `ollama list`

### "No module named 'xyz'"
- Install dependencies: `pip install -r requirements.txt`

### Documents not loading
- Check that files are in supported formats
- Ensure files are not corrupted
- Check console output for specific errors

### Vector database errors
- Clear and recreate: Click "Clear Docs" in the web interface
- Or manually delete the `chroma_db` folder

## 📝 Development

To modify or extend the chatbot:

1. **Change AI Model**: Edit `OLLAMA_MODEL` in the configuration
2. **Add File Types**: Extend the loaders in `ingest_documents.py`
3. **Customize UI**: Modify `templates/index.html` and `static/style.css`
4. **Adjust RAG Behavior**: Change chunk size, overlap, or retrieval count

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local AI model hosting
- [LangChain](https://langchain.com/) - Document processing and RAG framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings

---

Made with ❤️ using Python, Flask, and AI for Coding Questions

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
