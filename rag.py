# importing the relevant libraries
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import shutil

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Global Components
llm = None
vector_store = None


def initialize_components():
    """
    Initializes LLM and vector store if not already initialized
    """
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama3-70b-8192", temperature=0.9, max_tokens=500)

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


import shutil

def reset_vector_store():
    """
    Resets the vector store by clearing in-memory data and deleting persistent files.
    """
    global vector_store

    # Step 1: Delete all documents from current collection
    if vector_store is not None:
        try:
            vector_store._collection.delete(where={})  # Deletes all documents
        except Exception as e:
            print("Warning: Failed to delete from vector store:", e)

    # Step 2: Remove Chroma persistent files (safe on Windows)
    if VECTORSTORE_DIR.exists():
        try:
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print("Warning: Failed to clear vectorstore directory:", e)




def process_urls(urls):
    """
    Scrapes data from URLs and stores processed documents into Chroma DB.
    """
    yield "🧹 Resetting vector store..."
    reset_vector_store() # Clean slate

    yield "🔧 Initializing components..."
    initialize_components() # Reinitialize components (vector_store & llm)

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

    yield f"💾 Adding {len(docs)} documents to vector database..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

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

    answer, sources = generate_answer("create a summary on  30 year fixed mortgage rate")
    print(f"\n🧠 Answer: {answer}")
    print(f"📚 Sources: {sources}")
