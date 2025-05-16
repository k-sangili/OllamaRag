**OllamaRag**

A Retrieval-Augmented Generation (RAG) implementation using Ollama for local LLM inference combined with Pinecone for vector storage and document retrieval capabilities.
Overview
OllamaRag provides a simple yet powerful way to enhance local language models with the ability to retrieve and reference information from your documents. This implementation leverages Ollama for running local LLMs and implements RAG architecture to improve response quality by grounding model outputs in your own document collection, with Pinecone handling vector storage and retrieval.
Features

- 🤖 Use local LLMs through Ollama (default: deepseek-r1)
- 📚 Process and index multiple document formats (TXT, MD, PDF, DOCX, PPTX, code files)
- 🔍 Semantic search and retrieval with Pinecone vector database
- 💡 Context-aware responses using RetrievalQA
- 🖥️ Simple Gradio web interface for document processing and querying
- 📄 Metadata handling for source attribution

**Requirements**
- Python 3.8+
- Ollama installed and running locally
- Pinecone account (free tier works for testing)
- Required Python packages (see Installation section)

**Installation**
- Clone this repository:
bash
git clone https://github.com/k-sangili/OllamaRag.git
cd OllamaRag
- Install the required dependencies:
bash
pip install langchain langchain-community langchain-core sentence-transformers pinecone-client ollama-python python-dotenv gradio PyPDF2 python-docx python-pptx
- Install and run Ollama following the instructions at Ollama's official site.
- Set up environment variables:
bash
# Create a .env file with the following:
PINECONE_API_KEY=your_pinecone_api_key

PINECONE_ENVIRONMENT=your_pinecone_environment

PINECONE_INDEX_NAME=your_pinecone_index_name
- Create an appropriate Pinecone index with 768 dimensions (for all-mpnet-base-v2 embeddings).

**Usage**
- Run the application:
bash
python app.py

- Access the Gradio web interface at http://localhost:7860
- In the "Upload Documents" tab:
-- Enter the path to the folder containing your documents
-- Click "Process Documents" to index your files


In the "Ask Questions" tab:
- Enter your query about the documents
- Click "Ask" to get responses enhanced with information from your documents



Supported File Types
The system supports multiple file formats:

Text files (.txt)
Markdown files (.md)
Code files (.py, .js, .html, .css)
Data files (.json, .csv)
PDF files (.pdf)
Word documents (.docx)
PowerPoint presentations (.pptx)

**Configuration**
Key components that can be modified in the code:

- Ollama(model="deepseek-r1"): Change to any model available in your Ollama installation
- SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2"): Change embedding model if needed
- chunk_size=300, chunk_overlap=30: Adjust for different document chunking strategies
- search_kwargs={"k": 5}: Adjust the number of retrieved chunks

**Security Considerations**

- Your Pinecone API key and other credentials should be stored in a .env file (not committed to version control)
- Documents are processed locally before being embedded and stored in Pinecone
- Use proper access controls for your Pinecone index if handling sensitive information

**How It Works**

- Document Processing:
Documents are loaded from the specified folder
Text is extracted based on file type
Documents are split into smaller chunks
Each chunk is embedded using SentenceTransformer
Embeddings are stored in Pinecone with source metadata


- Query Processing:
User question is embedded using the same embedding model
Similar document chunks are retrieved from Pinecone
Local Ollama model generates a response based on the retrieved context
Both the answer and source references are returned

**Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Acknowledgments**
- Ollama for providing the local LLM runtime
- LangChain for the RAG implementation components
- Pinecone for the vector database
- Gradio for the web interface
- SentenceTransformers for the embedding model
