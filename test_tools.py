from rag_tool import rag_tool
from websearch_tool import websearch_tool

print("🧠 RAG Tool →")
print(rag_tool.run("¿Qué es un agente inteligente?"))

print("\n🌐 WebSearch Tool →")
print(websearch_tool.run("últimos avances en inteligencia artificial 2025"))