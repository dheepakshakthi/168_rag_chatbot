"""
168 - RAG Chatbot Flask Application
A beautiful web interface for the coding assistant chatbot
"""
from flask import Flask, render_template, request, jsonify, session
import os
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "smollm2:135m"
TOP_K_RESULTS = 4

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_PATH}. "
                "Please run ingest_documents.py first!"
            )
        
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )
    
    def retrieve_context(self, query):
        """Retrieve relevant context from vector store"""
        results = self.vectorstore.similarity_search_with_score(query, k=TOP_K_RESULTS)
        
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(
                f"[Source {i}: {os.path.basename(source)}, Page {page}]\n{doc.page_content}"
            )
            sources.append({
                'source': os.path.basename(source),
                'page': page,
                'score': float(score)
            })
        
        return "\n\n".join(context_parts), sources
    
    def generate_response(self, query, context):
        """Generate response using Ollama with RAG context"""
        prompt = f"""You are a helpful coding assistant. Answer the question based on the provided context from coding materials.

Context from coding materials:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so and provide general guidance
- Be concise and practical
- Include code examples when relevant
- Cite the source document when possible

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
        context, sources = self.retrieve_context(query)
        response = self.generate_response(query, context)
        return response, sources

# Initialize chatbot
try:
    chatbot = RAGChatbot()
    chatbot_ready = True
except Exception as e:
    chatbot_ready = False
    error_message = str(e)

@app.route('/')
def index():
    """Render the main page"""
    if not chatbot_ready:
        return render_template('error.html', error=error_message)
    return render_template('index.html')

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
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Empty message'
        })
    
    try:
        response, sources = chatbot.chat(query)
        return jsonify({
            'success': True,
            'response': response,
            'sources': sources
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
        print("🌐 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
    else:
        print(f"❌ Error: {error_message}")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
