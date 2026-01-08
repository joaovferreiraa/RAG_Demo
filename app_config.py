"""
Centralized configuration for RAG project.
"""
from pathlib import Path

class Config:
    """Configuration class for the RAG project."""
    
    # Paths
    DATA_PATH = "data"
    DB_PATH = "db"
    
    # Document processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    
    # Embedding
    EMBEDDING_MODEL = "text-embedding-3-small"
    COLLECTION_NAME = "state_of_data_2024"
    
    def __init__(self):
        """Create necessary directories."""
        Path(self.DATA_PATH).mkdir(exist_ok=True)
        Path(self.DB_PATH).mkdir(exist_ok=True)

# Create global instance
config = Config()