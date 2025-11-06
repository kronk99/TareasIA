import json
import numpy as np
import faiss
import pickle
from tqdm import tqdm

def build_faiss_index(input_file, index_file, metadata_file):
    embeddings = []
    metadatas = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Cargando {input_file}"):
            doc = json.loads(line)
            emb = np.array(doc["embedding"], dtype="float32")
            embeddings.append(emb)
            metadatas.append({
                "id": doc.get("id"),
                "chunk_id": doc.get("chunk_id"),
                "texto": doc.get("chunk"),
                "fuente": doc.get("fuente_archivo")
            })

    embeddings = np.vstack(embeddings)

    # Crear el índice FAISS (L2 distancia)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Guardar índice y metadata
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadatas, f)

    print(f"Index creado: {index_file}")
    print(f"Metadata guardada: {metadata_file}")
    print(f"Dimensiones del embedding: {dim}")
    print(f"Total de chunks indexados: {len(metadatas)}")

# Crear índices para ambas segmentaciones
build_faiss_index(
    "output/embeddings_parrafos.jsonl",
    "output/faiss_parrafos.index",
    "output/faiss_parrafos_meta.pkl"
)

build_faiss_index(
    "output/embeddings_sliding.jsonl",
    "output/faiss_sliding.index",
    "output/faiss_sliding_meta.pkl"
)