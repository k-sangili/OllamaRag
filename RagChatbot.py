from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os
import gradio as gr


# Install required packages:
# pip install langchain langchain-community langchain-core sentence-transformers pinecone-client ollama-python python-dotenv gradio


# Setup Environment Variables (Replace with your info)
load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')


# Validate credentials
if PINECONE_API_KEY == 'PINECONE_API_KEY':
    raise ValueError("Please set your Pinecone API key")
if PINECONE_ENVIRONMENT == 'PINECONE_ENVIRONMENT':
    raise ValueError("Please set your Pinecone environment")
if PINECONE_INDEX_NAME == 'PINECONE_INDEX_NAME':
    raise ValueError("Please set your Pinecone index name")


# 1. Load & Chunk Data (Example Text)
text = "This is a sample document. This document has multiple sentences. This sentence talks about RAG models. Retrieval Augmented Generation models are great. Another sentence about RAG. "
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_text(text)


# After chunking
print(f"Number of chunks created: {len(chunks)}")
print(f"Sample chunk: {chunks[0]}")


# 2. Embedding & Indexing
try:
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
   
    # Initialize Pinecone client
    pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
   
    # Check if index exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Index {PINECONE_INDEX_NAME} does not exist in your Pinecone account")
       
    # Create an index instance
    index = pc.Index(PINECONE_INDEX_NAME)


    # Prepare vectors for upsert with metadata
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        vectors.append({
            'id': str(i),
            'values': embedding,
            'metadata': {'text': chunk}  # Add metadata with the text
        })


    # Upsert the vectors
    index.upsert(vectors=vectors)
    # After embedding
    sample_embedding = embeddings.embed_query(chunks[0])
    print(f"Embedding dimension: {len(sample_embedding)}")


except Exception as e:
    print(f"Error during embedding and indexing: {str(e)}")
    raise
   
# 3. Retrieval & Generation function (moved out of the try block)
def ask_question(query):
    try:
      # Verify Ollama is running and model is available
      ollama_llm = Ollama(model="deepseek-r1")
     
      # Test the LLM with a simple query first
      test_response = ollama_llm.invoke("Test connection")
     
      # Use Langchain's Pinecone Vectorstore class for retrieval
      vectorstore = Pinecone(index, embeddings.embed_query, 'text')


      qa = RetrievalQA.from_chain_type(
          llm=ollama_llm,
          chain_type="stuff",
          retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant chunks
      )
     
      # Use invoke instead of run
      response = qa.invoke(query)
     
      # During retrieval
      retrieved_docs = vectorstore.as_retriever().get_relevant_documents(query)


      response_text = f"Response: {response['result']}\n\nRetrieved Documents:\n"
      for doc in retrieved_docs:
          response_text += f"- Document: {doc.page_content}\n"


      return response_text
   
    except Exception as e:
      return f"Error during retrieval and generation: {str(e)}"
   
# 4. Create Gradio Interface
iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=2, label="Enter your question:"),
    outputs=gr.Textbox(lines=10, label="Response"),
    title="RAG Chatbot with Ollama and Pinecone",
)


# 5. Launch the Interface
if __name__ == "__main__":
  iface.launch()
