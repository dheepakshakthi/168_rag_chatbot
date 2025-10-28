"""
168 - RAG Chatbot Flask Application
A beautiful web interface for a general-purpose AI chatbot with RAG capabilities
"""
from flask import Flask, render_template, request, jsonify, session
import os
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import secrets
import shutil
import time
import gc
import subprocess
import sys
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'data/'
app.config['HISTORY_FOLDER'] = 'chat_history/'

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "smollm2:1.7b"
TOP_K_RESULTS = 4

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTORY_FOLDER'], exist_ok=True)

# Check if we need to cleanup from previous session
def cleanup_on_startup():
    """Clean up chroma_db if flagged from previous session"""
    flag_file = "clear_docs_on_restart.flag"
    if os.path.exists(flag_file):
        print("🧹 Cleaning up from previous session...")
        try:
            if os.path.exists(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
                print("✅ Removed chroma_db")
            if os.path.exists(app.config['UPLOAD_FOLDER']):
                for file in os.listdir(app.config['UPLOAD_FOLDER']):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                print("✅ Cleared data folder")
            os.unlink(flag_file)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
            try:
                os.unlink(flag_file)
            except:
                pass

cleanup_on_startup()

# Chat history management functions
def save_chat_message(session_id, role, message, sources=None):
    """Save a chat message to history"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], f"{session_id}.json")
    
    try:
        # Load existing history or create new
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'messages': []
            }
        
        # Add new message
        message_data = {
            'role': role,
            'content': message,
            'timestamp': datetime.now().isoformat()
        }
        if sources:
            message_data['sources'] = sources
        
        history['messages'].append(message_data)
        history['updated_at'] = datetime.now().isoformat()
        
        # Save to file
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return False

def get_chat_history(session_id):
    """Get chat history for a session"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], f"{session_id}.json")
    
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return None

def list_chat_sessions():
    """List all chat sessions"""
    try:
        sessions = []
        for filename in os.listdir(app.config['HISTORY_FOLDER']):
            if filename.endswith('.json'):
                filepath = os.path.join(app.config['HISTORY_FOLDER'], filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                        
                    # Get first user message as preview
                    preview = "New conversation"
                    for msg in history.get('messages', []):
                        if msg['role'] == 'user':
                            preview = msg['content'][:50]
                            if len(msg['content']) > 50:
                                preview += "..."
                            break
                    
                    sessions.append({
                        'session_id': history['session_id'],
                        'created_at': history.get('created_at', ''),
                        'updated_at': history.get('updated_at', history.get('created_at', '')),
                        'message_count': len(history.get('messages', [])),
                        'preview': preview
                    })
                except:
                    continue
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        return sessions
    except Exception as e:
        print(f"Error listing chat sessions: {e}")
        return []

def delete_chat_session(session_id):
    """Delete a chat session"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], f"{session_id}.json")
    try:
        if os.path.exists(history_file):
            os.unlink(history_file)
            return True
        return False
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},  # Changed to CUDA for better performance
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.has_vectorstore = os.path.exists(CHROMA_PATH)
        self.vectorstore = None
        
        if self.has_vectorstore:
            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_PATH,
                    embedding_function=self.embeddings
                )
                print("✅ Vector store loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load vector store: {e}")
                self.has_vectorstore = False
    
    def retrieve_context(self, query):
        """Retrieve relevant context from vector store"""
        if not self.has_vectorstore or self.vectorstore is None:
            return None, []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=TOP_K_RESULTS)
            
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"[Source {i}: {os.path.basename(source)}]\n{doc.page_content}"
                )
                sources.append({
                    'source': os.path.basename(source),
                    'page': page,
                    'score': float(score)
                })
            
            return "\n\n".join(context_parts), sources
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return None, []
    
    def generate_response(self, query, context=None):
        """Generate response using Ollama with or without RAG context"""
        if context:
            prompt = f"""You are 168, a helpful AI assistant. Answer the question based on the provided context from the user's documents.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so and provide general guidance
- Be concise and helpful
- Cite the source document when possible

Answer:"""
        else:
            prompt = f"""You are 168, a helpful AI assistant. Answer the following question to the best of your ability.

Question: {query}

Answer:"""

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}\nMake sure Ollama is running and the model is available."
    
    def chat(self, query):
        """Main chat function"""
        if self.has_vectorstore:
            context, sources = self.retrieve_context(query)
        else:
            context, sources = None, []
        
        response = self.generate_response(query, context)
        return response, sources, self.has_vectorstore
    
    def close(self):
        """Properly close the vectorstore connection"""
        if self.vectorstore:
            try:
                # Try to reset the ChromaDB client to release file handles
                if hasattr(self.vectorstore, '_client'):
                    self.vectorstore._client.reset()
                if hasattr(self.vectorstore, '_collection'):
                    del self.vectorstore._collection
                del self.vectorstore
                self.vectorstore = None
                self.has_vectorstore = False
                # Force garbage collection to release resources
                gc.collect()
            except Exception as e:
                print(f"Warning during vectorstore closure: {e}")

# Initialize chatbot
try:
    chatbot = RAGChatbot()
    chatbot_ready = True
    print("✅ Chatbot initialized successfully")
except Exception as e:
    chatbot_ready = False
    error_message = str(e)
    print(f"❌ Error initializing chatbot: {e}")

def process_uploaded_files(files):
    """Process and ingest uploaded files"""
    saved_files = []
    
    for file in files:
        if file and file.filename:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_files.append(filename)
    
    if not saved_files:
        return False, "No valid files uploaded"
    
    # Run document ingestion
    try:
        from ingest_documents import load_documents, split_documents, create_vector_store
        
        documents = load_documents()
        if not documents:
            return False, "Could not load documents"
        
        chunks = split_documents(documents)
        create_vector_store(chunks)
        
        # Reinitialize chatbot to load new vector store
        global chatbot
        chatbot = RAGChatbot()
        
        return True, f"Successfully processed {len(saved_files)} file(s)"
    except Exception as e:
        return False, f"Error processing files: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    if not chatbot_ready:
        return render_template('error.html', error=error_message)
    has_docs = chatbot.has_vectorstore if chatbot_ready else False
    return render_template('index.html', has_documents=has_docs)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if 'files' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No files provided'
        })
    
    files = request.files.getlist('files')
    success, message = process_uploaded_files(files)
    
    return jsonify({
        'success': success,
        'message': message,
        'has_documents': chatbot.has_vectorstore if success else False
    })

def force_delete_directory(path, max_retries=5):
    """Force delete a directory on Windows, handling locked files"""
    import stat
    
    def handle_remove_readonly(func, path, exc):
        """Error handler for Windows readonly files"""
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                # Force garbage collection before attempting deletion
                gc.collect()
                time.sleep(0.3)
                
                # On Windows, try using rmdir /S /Q command as fallback
                if sys.platform == 'win32' and attempt > 2:
                    try:
                        abs_path = os.path.abspath(path)
                        subprocess.run(['cmd', '/c', 'rmdir', '/S', '/Q', abs_path], 
                                     check=False, capture_output=True, timeout=5)
                        if not os.path.exists(path):
                            return True
                    except:
                        pass
                
                # Try to delete with readonly handler
                shutil.rmtree(path, onerror=handle_remove_readonly)
                return True
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            else:
                # Last attempt: try to rename instead
                try:
                    import uuid
                    backup_name = f"{path}_to_delete_{uuid.uuid4().hex[:8]}"
                    os.rename(path, backup_name)
                    print(f"⚠️ Moved {path} to {backup_name}")
                    return False
                except:
                    raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
    return False

@app.route('/clear_documents', methods=['POST'])
def clear_documents():
    """Clear all uploaded documents and vector store"""
    global chatbot
    
    try:
        # Create a flag file to indicate documents should be cleared on restart
        flag_file = "clear_docs_on_restart.flag"
        
        # Step 1: Close the vectorstore connection
        if chatbot:
            print("Closing vectorstore connection...")
            chatbot.close()
        
        # Step 2: Delete the chatbot object completely
        chatbot = None
        gc.collect()  # Force garbage collection
        time.sleep(1)  # Give Windows time to release file handles
        
        # Step 3: Try to delete immediately
        deleted_successfully = False
        if os.path.exists(CHROMA_PATH):
            print(f"Attempting to delete {CHROMA_PATH}...")
            try:
                # Try direct deletion first
                shutil.rmtree(CHROMA_PATH)
                deleted_successfully = True
                print("✅ Successfully deleted chroma_db")
            except Exception as e:
                print(f"⚠️ Could not delete immediately: {e}")
                # Create flag file for restart cleanup
                with open(flag_file, 'w') as f:
                    f.write("1")
        
        # Step 4: Clear data directory
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"Warning: Could not delete {file_path}: {e}")
        
        # Step 5: Reinitialize chatbot
        print("Reinitializing chatbot...")
        chatbot = RAGChatbot()
        
        if deleted_successfully:
            message = 'Documents cleared successfully'
        else:
            message = 'Please restart the application to complete document clearing (close and run python app.py again)'
        
        return jsonify({
            'success': True,
            'message': message,
            'needs_restart': not deleted_successfully
        })
    except Exception as e:
        error_msg = str(e)
        print(f"Error in clear_documents: {error_msg}")
        
        # Try to reinitialize chatbot anyway
        try:
            chatbot = RAGChatbot()
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': 'Could not delete vector database. Please restart the app (close terminal and run "python app.py" again).'
        })

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    if not chatbot_ready:
        return jsonify({
            'success': False,
            'error': 'Chatbot is not initialized'
        })
    
    data = request.json
    query = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Empty message'
        })
    
    try:
        # Save user message
        save_chat_message(session_id, 'user', query)
        
        # Get response from chatbot
        response, sources, has_rag = chatbot.chat(query)
        
        # Save assistant response
        save_chat_message(session_id, 'assistant', response, sources)
        
        return jsonify({
            'success': True,
            'response': response,
            'sources': sources,
            'has_rag': has_rag,
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history/sessions', methods=['GET'])
def get_sessions():
    """Get list of all chat sessions"""
    try:
        sessions = list_chat_sessions()
        return jsonify({
            'success': True,
            'sessions': sessions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get chat history for a specific session"""
    try:
        history = get_chat_history(session_id)
        if history:
            return jsonify({
                'success': True,
                'history': history
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    try:
        if delete_chat_session(session_id):
            return jsonify({
                'success': True,
                'message': 'Session deleted'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/files', methods=['GET'])
def list_files():
    """List uploaded files"""
    try:
        files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(filepath):
                    file_stat = os.stat(filepath)
                    files.append({
                        'name': filename,
                        'size': file_stat.st_size,
                        'uploaded_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
        
        # Sort by upload time descending
        files.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return jsonify({
            'success': True,
            'files': files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a specific uploaded file"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            os.unlink(filepath)
            return jsonify({
                'success': True,
                'message': f'File {filename} deleted. Please re-upload files to rebuild the database.'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'File not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear', methods=['POST'])
def clear():
    """Clear chat history"""
    return jsonify({'success': True})

if __name__ == '__main__':
    print("=" * 70)
    print("168 - RAG Chatbot Web Application")
    print("=" * 70)
    if chatbot_ready:
        print("✅ Chatbot initialized successfully!")
        if chatbot.has_vectorstore:
            print("📚 Vector store loaded - RAG mode enabled")
        else:
            print("💬 No documents loaded - General chat mode")
        print("🌐 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
    else:
        print(f"❌ Error: {error_message}")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
