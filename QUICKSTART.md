# Quick Start Guide - 168 RAG Chatbot

## 🚀 Get Started in 3 Steps

### Step 1: Make Sure Ollama is Running

```bash
# Check if Ollama is installed
ollama --version

# If not installed, download from: https://ollama.ai/

# Pull the model (if you haven't already)
ollama pull smollm2:135m

# Start Ollama (if not already running)
ollama serve
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd d:\168_rag_chatbot

# Install required packages
pip install -r requirements.txt
```

### Step 3: Launch the Application

```bash
# Start the web app
python app.py
```

Open your browser and go to: **http://localhost:5000**

---

## 💡 Two Ways to Use

### Option A: General Chat Mode (No Documents)
1. Open the app
2. Start typing questions directly
3. Get AI-powered answers on any topic
4. No file upload needed!

**Example Questions:**
- "What is Python?"
- "Explain quantum computing"
- "How do I cook pasta?"

### Option B: RAG Mode (With Your Documents)
1. Click **"Upload Files"** button
2. Select your files (PDF, HTML, code, etc.)
3. Wait for processing (you'll see a success notification)
4. Ask questions about your documents!

**Example Questions:**
- "Summarize the main points from the PDF"
- "What functions are defined in the code files?"
- "Explain the concept discussed in page 5"

---

## 📤 Uploading Files

### Supported File Types:
- **Documents**: PDF, HTML, TXT, MD
- **Code**: Python (.py), Java (.java), JavaScript (.js), TypeScript (.ts)
- **More Code**: C/C++ (.c, .cpp, .h), C# (.cs), Ruby (.rb), Go (.go), Rust (.rs), PHP (.php), Swift (.swift), Kotlin (.kt)

### How to Upload:
1. Click the **"Upload Files"** button in the header
2. Select one or multiple files from your computer
3. Wait for the processing notification
4. 🟢 **RAG Mode Active** - Now you can ask questions!

### File Size Limit:
- Maximum 50MB per upload session
- Upload as many files as needed

---

## 🔄 Switching Modes

### From General to RAG Mode:
- Just upload files
- Mode switches automatically
- Header shows: **(RAG Mode)**

### From RAG to General Mode:
- Click **"Clear Docs"** button
- Confirm the action
- All documents and vector database are deleted
- Mode switches to: **(General Mode)**

---

## 🎮 Interface Controls

### Header Buttons:
- **📤 Upload Files**: Add documents to enable RAG mode
- **🗑️ Clear Docs**: Remove all documents and return to general mode
- **🗑️ Clear Chat**: Clear conversation history (keeps documents)

### Chat Input:
- **Enter**: Send message
- **Shift + Enter**: New line in message

---

## 🔍 Understanding Responses

### In General Mode:
```
You: What is machine learning?
🤖 Assistant: [General AI knowledge response]
```

### In RAG Mode:
```
You: What does the document say about machine learning?
🤖 Assistant: [Response based on your documents]
📚 Sources:
  1. my-document.pdf (Page 5) - Relevance: 0.85
  2. notes.txt - Relevance: 0.72
```

---

## 🛠️ Troubleshooting

### "Connection Error"
**Problem**: Can't connect to Ollama
**Solution**: 
```bash
# Start Ollama in a terminal
ollama serve
```

### "Model Not Found"
**Problem**: SmolLM2 model not installed
**Solution**:
```bash
ollama pull smollm2:135m
```

### Files Not Processing
**Problem**: Upload fails or takes too long
**Solutions**:
- Check file size (max 50MB total)
- Ensure file format is supported
- Check console output for errors
- Try uploading fewer files at once

### Slow Responses
**Problem**: AI takes long to respond
**Solutions**:
- First response is always slower (model loading)
- Large documents take longer to process
- Consider using a smaller number of files
- Check CPU usage (embeddings use CPU)

---

## 💻 Command Line Interface

Prefer terminal? Use the CLI version:

```bash
python chatbot.py
```

### CLI Commands:
- Type your questions normally
- **`sources`** - View sources from last RAG response
- **`quit`** or **`exit`** - Exit the program

### CLI Document Setup:
1. Place files in the `data/` folder
2. Run: `python ingest_documents.py`
3. Run: `python chatbot.py`

---

## 📊 Example Workflow

### Research Assistant Example:
```
1. Upload research papers (PDFs)
2. Ask: "What are the main findings?"
3. Ask: "Compare the methodologies used"
4. Ask: "What are the limitations mentioned?"
```

### Code Documentation Example:
```
1. Upload your Python/Java files
2. Ask: "What does the main function do?"
3. Ask: "List all the classes defined"
4. Ask: "Explain the error handling approach"
```

### Learning Assistant Example:
```
1. Upload course materials (PDFs, HTMLs)
2. Ask: "Explain the key concepts from chapter 3"
3. Ask: "Give me examples of X"
4. Ask: "What's the difference between Y and Z?"
```

---

## 🎯 Best Practices

### For Best Results:
1. **Upload relevant documents** - Only upload files related to your questions
2. **Be specific** - Ask clear, focused questions
3. **Reference sources** - Ask about specific documents or sections
4. **Iterate** - Ask follow-up questions for deeper understanding

### Document Tips:
- Use clear, well-formatted documents
- Text-heavy PDFs work better than image-heavy ones
- Organize related documents together
- Remove irrelevant files for better results

### Question Tips:
- ✅ "What does the document say about X?"
- ✅ "Summarize the section on Y"
- ✅ "Compare X and Y from the files"
- ❌ "Tell me everything" (too broad)

---

## 🆘 Need Help?

### Check the Documentation:
- **README.md** - Comprehensive guide
- **CHANGES.md** - List of all features and changes

### Common Issues:
1. **Ollama not running** → Start it with `ollama serve`
2. **Model missing** → Download with `ollama pull smollm2:135m`
3. **Dependencies missing** → Install with `pip install -r requirements.txt`
4. **Port in use** → Change port in app.py or close other apps using port 5000

---

## 🎉 You're Ready!

That's it! You now have a powerful AI assistant that can:
- Chat about anything (General Mode)
- Answer questions from your documents (RAG Mode)
- Process multiple file formats
- Provide source citations

**Enjoy your AI assistant!** 🤖✨
