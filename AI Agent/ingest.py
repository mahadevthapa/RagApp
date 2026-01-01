import chromadb
from sentence_transformers import SentenceTransformer

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#load client
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="knowledge_base")

documents =[
    "Python is widely used in artificial intelligence for machine learning and deep learning.",
    "Libraries such as NumPy, TensorFlow, PyTorch, and scikit-learn support AI development.",
    "AI systems rely on data processing, model training, and inference pipelines."
]


#chunking
def chunk_text(text, size=400, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

all_chunks = []
ids = []

for i, doc in enumerate(documents):
    chunks = chunk_text(doc)
    for j, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        ids.append(f"doc_{i}_{j}")
        

embeddings = model.encode(all_chunks).tolist()

collection.add(documents = all_chunks,
               embeddings = embeddings,
               ids = ids
               )

print("Ingestion Complete")
print("Total chunks: ", collection.count())