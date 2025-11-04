"""
RAG Chatbot with Ollama
Interactive chatbot that can answer questions using RAG (if documents are loaded) or general mode
"""
import os
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "smollm2:1.7b"
TOP_K_RESULTS = 4

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot"""
        print("Initializing RAG Chatbot...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Check if vector store exists
        self.has_vectorstore = os.path.exists(CHROMA_PATH)
        self.vectorstore = None
        
        if self.has_vectorstore:
            try:
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_PATH,
                    embedding_function=self.embeddings
                )
                print("✅ Vector store loaded - RAG mode enabled")
            except Exception as e:
                print(f"⚠️ Warning: Could not load vector store: {e}")
                self.has_vectorstore = False
                print("💬 Running in general chat mode")
        else:
            print("💬 No documents found - Running in general chat mode")
        
        print("Chatbot initialized successfully!")
    
    def retrieve_context(self, query):
        """Retrieve relevant context from vector store"""
        if not self.has_vectorstore or self.vectorstore is None:
            return None, []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=TOP_K_RESULTS)
            
            # Format context
            context_parts = []
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                context_parts.append(
                    f"[Source {i}: {os.path.basename(source)}]\n{doc.page_content}"
                )
            
            return "\n\n".join(context_parts), results
        except Exception as e:
            print(f"⚠️ Error retrieving context: {e}")
            return None, []
    
    def generate_response(self, query, context=None):
        """Generate response using Ollama with or without RAG context"""
        if context:
            prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context from the user's documents.

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
            prompt = f"""You are a helpful AI assistant. Answer the following question to the best of your ability.

Question: {query}

Answer:"""

        try:
            # Call Ollama
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
            print("\n🔍 Retrieving relevant information...")
            context, results = self.retrieve_context(query)
            
            if results:
                print(f"📚 Found {len(results)} relevant document sections")
        else:
            context, results = None, []
            print("\n💬 Answering in general mode...")
        
        print("\n🤖 Generating response...\n")
        response = self.generate_response(query, context)
        
        return response, results

def main():
    """Main chatbot interface"""
    print("=" * 70)
    print("RAG-Based AI Chatbot with SmolLM2:1.7b")
    print("=" * 70)
    print("\nThis chatbot can answer questions based on your uploaded documents")
    print("or chat with you in general mode if no documents are loaded.")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'sources' to see the last retrieved sources (RAG mode only)")
    print("=" * 70)
    
    try:
        chatbot = RAGChatbot()
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        return
    
    last_results = None
    
    print("\n✅ Chatbot ready! Ask me anything.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'sources':
                if chatbot.has_vectorstore and last_results:
                    print("\n📚 Sources from last query:")
                    for i, (doc, score) in enumerate(last_results, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        print(f"  {i}. {os.path.basename(source)} (Page {page}) - Similarity: {score:.3f}")
                else:
                    print("\n❌ No sources available (either no previous query or running in general mode).")
                continue
            
            response, last_results = chatbot.chat(query)
            print(f"\n🤖 Assistant:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
