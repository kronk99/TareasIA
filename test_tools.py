from rag_tool import rag_tool
from websearch_tool import websearch_tool
from rag_tool_sliding import rag_tool_sliding
#print(" Probando WebSearch Tool...")
#query = "últimos avances en inteligencia artificial 2025"
#print(websearch_tool.run(query))

#print(" RAG Tool →")
#print(rag_tool.run("¿Qué es time masking?"))

print(" RAG Tool sliding →")
print(rag_tool_sliding.run("¿Qué es Time Masking?"))

#print("\n WebSearch Tool →")
#print(websearch_tool.run("últimos avances en inteligencia artificial 2025"))

#definicion del prompt base : el prompt base tendrá una memoria corto plazo de no más
#de 3 respuestas anteriores.
