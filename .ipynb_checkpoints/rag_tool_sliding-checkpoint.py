import faiss
import numpy as np
import pickle
from openai import OpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import re
client = OpenAI()

def load_faiss_index(index_path, metadata_path):
    # Lee el índice y la metadata desde disco
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_vector_db(query, index, metadata, model="text-embedding-3-small", top_k=3):
    """Dado un texto de consulta, obtiene su embedding y busca los top_k
    fragmentos más similares en la base de datos vectorial."""
    # Generar embedding para la consulta
    response = client.embeddings.create(model=model, input=query)
    query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    # Buscar en el índice FAISS
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(metadata):
            meta = metadata[idx]
            # Construir la fuente con documento y autor
            fuente = f"{meta['documento']} — {meta.get('autor', '')}".strip(" —")
            results.append({
                "chunk_id": meta.get("chunk_id", ""),  # linea para el sorting
                "texto": meta["chunk"],                # contenido del párrafo recuperado
                "fuente": fuente,                      # documento — autor
                "distancia": float(dist)
            })
    return results

# Cargar el índice y la metadata (ejemplo para párrafos)
index, metadata = load_faiss_index(
    "output/faiss_sliding.index",
    "output/faiss_sliding_meta.pkl"
)

class RAGToolInput(BaseModel):
    query: str = Field(..., description="Consulta del usuario para buscar en la base vectorial")

def rag_tool_function(query: str):
    # Recupera los fragmentos más relevantes y concatena sus textos
    results = search_vector_db(query, index, metadata)
    #return "\n".join([r["texto"] for r in results])
    if not results:
        return "⚠️ No se encontraron resultados relevantes."
    
    #ordena los resultados por chunk id para un mejor contexto al orquestador
    results.sort(
        key=lambda r: int(re.search(r"[sp](\d+)$", r["chunk_id"]).group(1)) if re.search(r"[sp](\d+)$", r["chunk_id"]) else 0
    )
    

    # Extraer el texto principal (contenido de los chunks)
    respuesta = "\n".join([r["texto"] for r in results])

    # Construir las referencias con fuente (documento — autor)
    referencias = "\n".join([
        f"- {r['fuente']}" for r in results
    ])

    # Retornar un único string formateado (compatible con .run)
    return (
        f" Respuesta:\n{respuesta}\n\n"
        f" Referencias:\n{referencias}"
    )

# Definición del RAG Tool para LangChain/Core
rag_tool_sliding = StructuredTool(
    name="RAG_Tool",
    func=rag_tool_function,
    description="Extrae información contextual desde la base de datos vectorial (RAG).",
    args_schema=RAGToolInput
)
