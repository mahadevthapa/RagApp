import chromadb
from sentence_transformers import SentenceTransformer

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#client
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection(name="knowledge_base")

def rag_search(query:str, top_k: int=5) -> str:
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings = embedding,
        n_results = top_k
    )
    
    docs = results["documents"][0]
    return "\n".join(docs) if docs else ""
        
         
        
        