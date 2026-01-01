import chromadb
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_store")
memory = client.get_or_create_collection(name="agent_memory")


def store_memory(text:str):
    embedding = model.encode([text]).tolist()
    memory.add(
        documents = [text],
        embeddings = embedding,
        ids=[str(time.time())]
    )
    
def recall_memory(query:str, top_k: int = 3) ->str:
    embedding = model.encode([query]).tolist()
    results = memory.query(
        query_embeddings = embedding,
        n_results = top_k
    )
    
    docs = results["documents"][0]
    return "\n".join(docs) if docs else ""