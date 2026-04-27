import chromadb
from chromadb.config import Settings

settings = Settings(persist_directory="chroma_data")

client = chromadb.Client(settings)

collection = client.get_or_create_collection("stock_data")

# Retrieve all documents in the collection
all_docs = collection.get(include=['documents', 'metadatas', 'embeddings', 'ids'])
print("Stored IDs:", all_docs['ids'])
print("Metadatas:", all_docs['metadatas'])
print("Documents:", all_docs['documents'])