import os
import json
from groq import Groq

from rag import rag_search
from memory import store_memory, recall_memory

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search internal documents for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(question:str):
    past_memory = recall_memory(question)
    
    messages = [
        {
            "role":"system",
            "content":
                "You are a research agent.\n"
                "Use tools only when information is needed.\n"
                "After using a tool once, answer the question.\n"
                "If information is missing, say 'I don't know'."
        }
    ]
    
    if past_memory:
        messages.append({
            "role":"system",
            "content":f"Relevant past memory:\n{past_memory}"
        })
        
    messages.append({"role":"user", "content":question})
    
    tool_used = False
    
    for _ in range(5):
        response = llm.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages = messages,
            tools = tools,
            tool_choice = "auto",
            temperature = 0.2
        )
        
    msg = response.choices[0].message
    
    if msg.tool_calls and not tool_used:
        tool_used = True
        
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        
        result = rag_search(**args)
        
        messages.append({
            "role":"tool",
            "tool_call_id":tool_call.id,
            "name":tool_call.function.name,
            "content":result
        })
        
        messages.append({
            "role":"user",
            "content":"Using the retrieved information, answer the quesiton."
        })
    else:
        answer = msg.content or "I don't know."
        print("\nFinal answer:", answer)
        store_memory(f"Q: {question}\nA:{answer}")
        return
    print("Stopped: step limit reached")
    
run_agent("What is the role of Python in AI?")
    

        
        
        
        
