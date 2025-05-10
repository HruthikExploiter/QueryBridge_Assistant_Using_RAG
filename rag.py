# importing the relevant libraries
from uuid import uuid4
import streamlit as st
from pathlib import Path
import shutil

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Get API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Global Components
llm = None
vector_store = None


def initialize_components():
    """
    Initializes LLM and vector store, loading FAISS from disk if available
    """
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-70b-8192",
            temperature=0.9,
            max_tokens=500
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
            # Load existing index
            vector_store = FAISS.load_local(str(VECTORSTORE_DIR), embeddings)
        else:
            # Create empty vector store
            vector_store = FAISS.from_documents([], embedding=embeddings)


def reset_vector_store():
    """
    Resets the FAISS vector store by deleting persistent index files.
    """
    global vector_store
    vector_store = None  # Clear in-memory reference

    # Remove FAISS persistent files
    if VECTORSTORE_DIR.exists():
        try:
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print("Warning: Failed to reset vectorstore directory:", e)


def process_urls(urls):
    """
    Scrapes data from URLs and stores processed documents into FAISS.
    """
    yield "🧹 Resetting vector store..."
    reset_vector_store()

    yield "🔧 Initializing components..."
    initialize_components()

    yield "📥 Loading data from URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "✂️ Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)

    yield f"💾 Adding {len(docs)} documents to FAISS database..."
    vector_store.add_documents(docs)

    # Save the FAISS index to disk
    vector_store.save_local(str(VECTORSTORE_DIR))

    yield "✅ Vector DB update complete!"


def generate_answer(query):
    """
    Answers a query using the retrieval-augmented generation chain.
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized.")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)
    return result.get("answer", ""), result.get("sources", "")


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for update in process_urls(urls):
        print(update)

    answer, sources = generate_answer("create a summary on 30 year fixed mortgage rate")
    print(f"\n🧠 Answer: {answer}")
    print(f"📚 Sources: {sources}")
