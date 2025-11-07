import faiss
import numpy as np
import pickle
from openai import OpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
client = OpenAI()

def load_faiss_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_vector_db(query, index, metadata, model="text-embedding-3-small", top_k=3):
    response = client.embeddings.create(model=model, input=query)
    query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(metadata):
            results.append({
                "texto": metadata[idx]["texto"],
                "fuente": metadata[idx]["fuente"],
                "distancia": float(dist)
            })
    return results

# Carga el índice que prefieras (puedes alternar entre parrafos o sliding)
index, metadata = load_faiss_index(
    "output/faiss_parrafos.index", 
    "output/faiss_parrafos_meta.pkl"
)
class RAGToolInput(BaseModel):
    query: str = Field(..., description="Consulta del usuario para buscar en la base vectorial")

def rag_tool_function(query: str):
    results = search_vector_db(query, index, metadata)
    return "\n".join([r["texto"] for r in results])

rag_tool = StructuredTool(
    name="RAG_Tool",
    func=rag_tool_function,
    description="Extrae información contextual desde la base de datos vectorial (RAG).",
    args_schema=RAGToolInput
)


#