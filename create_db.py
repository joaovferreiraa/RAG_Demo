from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from app_config import config
from pathlib import Path
import logging

# Configuration
load_dotenv() # Load OpenAI Key
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_db():
    """Complete pipeline to create vectorized database."""

    logger.info("Starting database creation...")

    documents = load_documents()
    logger.info(f"✓ {len(documents)} documents loaded")

    chunks = divide_chunks(documents)
    logger.info(f"✓ {len(chunks)} chunks created")

    vectorize_chunks(chunks)
    logger.info("✓ Successfully vectorized database!")

def load_documents():
    """Load PDFs from specified directory."""
    loader = PyPDFDirectoryLoader(config.DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def divide_chunks(documents):
    """Separate documents in chunks, enriched with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)
    
    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "source_file": Path(chunk.metadata.get("source", "")).name,
            "chunk_size": len(chunk.page_content)
        })

    return chunks

def vectorize_chunks(chunks):

    """Create vectorized database with OpenAI embeddings using FAISS."""
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    db = FAISS.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        )
    # Save to disk
    db.save_local(config.DB_PATH)

    return db

if __name__ == "__main__":
    create_db()