from rag_tool import rag_tool
from websearch_tool import websearch_tool

#print("🌐 Probando WebSearch Tool...")
#query = "últimos avances en inteligencia artificial 2025"
#print(websearch_tool.run(query))

print("🧠 RAG Tool →")
print(rag_tool.run("¿Por quién fue creado el modelo LeNet-5 clasico?"))

#print("\n🌐 WebSearch Tool →")
#print(websearch_tool.run("últimos avances en inteligencia artificial 2025"))

#definicion del prompt base : el prompt base tendrá una memoria corto plazo de no más
#de 3 respuestas anteriores.
