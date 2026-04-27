# utils.py

from sentence_transformers import SentenceTransformer

# Initialize Embeddings model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    return embedding_model.encode([text])[0].tolist()
