import json
import unicodedata
import re

INPUT_FILE = "output/documents.jsonl"
OUT_PARAGRAPH = "output/documents_parrafos.jsonl"
OUT_SLIDING = "output/documents_sliding.jsonl"

CHUNK_SIZE = 300  # palabras por chunk
OVERLAP = 100     # solapamiento

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text)  # espacios múltiples → uno
    text = re.sub(r'[^\x00-\x7Fñáéíóúü°°%()¡!¿?.,;:0-9a-z ]', '', text)
    return text.strip()

def split_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if len(p.strip().split()) > 10]

def sliding_window(words, size, overlap):
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        if len(chunk) < 30:
            break
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUT_PARAGRAPH, "w", encoding="utf-8") as out_par, \
     open(OUT_SLIDING, "w", encoding="utf-8") as out_slide:

    for line in infile:
        doc = json.loads(line)
        text = normalize_text(doc["contenido"])

        # --- Párrafos
        for idx, p in enumerate(split_paragraphs(text)):
            record = {**doc, "chunk_id": f"{doc['id']}_p{idx}", "chunk": p}
            out_par.write(json.dumps(record, ensure_ascii=False) + "\n")

        # --- Sliding window
        words = text.split()
        chunks = sliding_window(words, CHUNK_SIZE, OVERLAP)
        for idx, c in enumerate(chunks):
            record = {**doc, "chunk_id": f"{doc['id']}_s{idx}", "chunk": c}
            out_slide.write(json.dumps(record, ensure_ascii=False) + "\n")