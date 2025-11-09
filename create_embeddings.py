import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def embed_file(input_file, output_file, model="text-embedding-3-small"):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc=f"Embedding {input_file}"):
            doc = json.loads(line)
            # Usamos el texto del párrafo
            text = doc["chunk"]

            # Solicitar el embedding al modelo
            response = client.embeddings.create(
                model=model,
                input=text
            )
            # El embedding es una lista de 1536 floats
            embedding = response.data[0].embedding

            # Añadirlo al dict
            doc["embedding"] = embedding

            # Escribir el doc con el embedding
            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")

# Generar embeddings para el preprocesado por párrafos
embed_file("output/documents_parrafos.jsonl", "output/embeddings_parrafos.jsonl")
#generar embeddings para el preprocesado por slidings
embed_file("output/documents_sliding.jsonl", "output/embeddings_sliding.jsonl")
