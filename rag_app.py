import chromadb
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

#initialize components
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection(name="knowledge_base")

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rag_answer(question, top_k = 3):
    #retrieve
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = top_k
    )
    
    context = "\n".join(results["documents"][0])
    
    prompt = f"""
    You are a helpful assistant.
    Answer the question using only the context below.
    If the answer is not in the context, say "I don't know."
    
    Context:{context}
    
    Question:{question}
    """
    
    #Generate
    response = llm.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [{"role":"user", "content":prompt}],
        temperature = 0.2,
        max_tokens = 200
    )
    
    return response.choices[0].message.content

#Test
question = "How is python used in AI?"
print(rag_answer(question))