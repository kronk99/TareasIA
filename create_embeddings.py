import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def embed_file(input_file, output_file, model="text-embedding-3-small"):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc=f"Embedding {input_file}"):
            doc = json.loads(line)
            text = doc["chunk"]

            # Pide el embedding al modelo
            response = client.embeddings.create(
                model=model,
                input=text
            )

            embedding = response.data[0].embedding
            doc["embedding"] = embedding

            outfile.write(json.dumps(doc) + "\n")

embed_file("output/documents_parrafos.jsonl", "output/embeddings_parrafos.jsonl")
embed_file("output/documents_sliding.jsonl", "output/embeddings_sliding.jsonl")

print("Embeddings generados con éxito")