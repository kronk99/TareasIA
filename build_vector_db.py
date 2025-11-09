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

            # Convertir embedding a numpy array
            emb = np.array(doc["embedding"], dtype="float32")
            embeddings.append(emb)

            # Guardar metadata relevante
            metadatas.append({
                "chunk_id": doc["chunk_id"],
                "chunk": doc["chunk"],
                "autor": doc.get("autor"),
                "documento": doc.get("documento")
            })

    # Convertir a matriz 2D
    embeddings = np.vstack(embeddings)

    # Crear índice FAISS para distancias L2
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

# Crear el índice para la segmentación por párrafos

build_faiss_index(
    "output/embeddings_parrafos.jsonl",
    "output/faiss_parrafos.index",
    "output/faiss_parrafos_meta.pkl"
)
# Crear el índice para la segmentación para slidings
build_faiss_index(
    "output/embeddings_sliding.jsonl",
    "output/faiss_sliding.index",
    "output/faiss_sliding_meta.pkl"
)
