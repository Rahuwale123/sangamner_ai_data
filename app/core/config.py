import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant Configuration
# Default to hosted instance unless overridden by env
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "sangmaner_data")

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Vector dimensions for the embeding model
VECTOR_SIZE = 384

# (AI mode removed) No Gemini configuration required

# Semantic search/ranking configuration (tunable via env vars)
# Minimum cosine similarity required to keep a hit (0..1)
SEMANTIC_SCORE_THRESHOLD = float(os.getenv("SEMANTIC_SCORE_THRESHOLD", "0.35"))

# Weights for blended scoring when geo context is used
# Note: When geo is not applicable, only semantic score is used
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.8"))
GEO_WEIGHT = float(os.getenv("GEO_WEIGHT", "0.2"))