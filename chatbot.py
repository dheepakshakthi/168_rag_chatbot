"""
RAG Chatbot with Ollama
Interactive chatbot that answers coding questions using RAG and SmolLM2:1.7b
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
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_PATH}. "
                "Please run ingest_documents.py first!"
            )
        
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )
        print("Chatbot initialized successfully!")
    
    def retrieve_context(self, query):
        """Retrieve relevant context from vector store"""
        results = self.vectorstore.similarity_search_with_score(query, k=TOP_K_RESULTS)
        
        # Format context
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(
                f"[Source {i}: {os.path.basename(source)}, Page {page}]\n{doc.page_content}"
            )
        
        return "\n\n".join(context_parts), results
    
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
        print("\n🔍 Retrieving relevant information...")
        context, results = self.retrieve_context(query)
        
        print(f"📚 Found {len(results)} relevant document sections")
        
        print("\n🤖 Generating response...\n")
        response = self.generate_response(query, context)
        
        return response, results

def main():
    """Main chatbot interface"""
    print("=" * 70)
    print("RAG-Based Coding Chatbot with SmolLM2:1.7b")
    print("=" * 70)
    print("\nThis chatbot can answer questions about Python and Java programming")
    print("based on the learning materials you've provided.")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'sources' to see the last retrieved sources")
    print("=" * 70)
    
    try:
        chatbot = RAGChatbot()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        return
    
    last_results = None
    
    print("\n✅ Chatbot ready! Ask me anything about coding.\n")
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Happy coding!")
                break
            
            if query.lower() == 'sources':
                if last_results:
                    print("\n📚 Sources from last query:")
                    for i, (doc, score) in enumerate(last_results, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        print(f"  {i}. {os.path.basename(source)} (Page {page}) - Similarity: {score:.3f}")
                else:
                    print("\n❌ No previous query to show sources for.")
                continue
            
            response, last_results = chatbot.chat(query)
            print(f"\n🤖 Assistant:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy coding!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
