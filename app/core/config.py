import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "sangmaner_data")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector dimensions for the embedding model
VECTOR_SIZE = 384

# Gemini AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAE0oIus2UKhuI3MdDDbvqlZB03c-WOvKs")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")
