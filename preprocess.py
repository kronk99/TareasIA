import json
import unicodedata
import re

INPUT_FILE = "output/prueba.jsonl"
OUT_PARAGRAPH = "output/documents_parrafos.jsonl"
OUT_SLIDING  = "output/documents_sliding.jsonl"

CHUNK_SIZE = 300   # palabras por chunk para sliding window
OVERLAP    = 100   # solapamiento

def normalize_text(text: str) -> str:
    """
    Convierte texto a minúsculas y normaliza caracteres sin eliminar los saltos de línea.
    Sustituye secuencias de espacios y tabulaciones, pero conserva los \n para poder
    segmentar por párrafos más adelante.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    # Unificar \r\n y \r en \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Sustituir múltiples espacios o tabs por un solo espacio (no toca \n)
    text = re.sub(r'[ \t]+', ' ', text)
    # Eliminar caracteres no permitidos pero dejar \n
    text = re.sub(r'[^\x00-\x7Fñáéíóúü°%()¡!¿?.,;:0-9a-z\n ]', '', text)
    return text.strip()

def split_paragraphs(text: str):
    """
    Cada línea del texto (separada por '\n') será tratada como un párrafo
    si contiene más de 10 palabras. Esto permite conservar la estructura
    original de los apuntes.
    """
    # Dividir por saltos de línea individuales
    lines = text.split('\n')
    # Solo tomar líneas con contenido suficiente
    return [ln.strip() for ln in lines if len(ln.strip().split()) > 10]

def sliding_window(words, size, overlap):
    """Permanece sin cambios; genera chunks solapados de tamaño 'size'."""
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        if len(chunk) < 30:
            break
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

# Abrir archivos y procesar cada documento
with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUT_PARAGRAPH, "w", encoding="utf-8") as out_par, \
     open(OUT_SLIDING,  "w", encoding="utf-8") as out_slide:

    for line in infile:
        doc = json.loads(line)
        text = normalize_text(doc["contenido"])

        # --- Segmentación por párrafos (cada salto de línea)
        for idx, p in enumerate(split_paragraphs(text)):
            record = {
                **doc,
                "chunk_id": f"{doc['id']}_p{idx}",
                "chunk": p
            }
            out_par.write(json.dumps(record, ensure_ascii=False) + "\n")

        # --- Segmentación por sliding window (sin cambios)
        words = text.split()
        chunks = sliding_window(words, CHUNK_SIZE, OVERLAP)
        for idx, c in enumerate(chunks):
            record = {
                **doc,
                "chunk_id": f"{doc['id']}_s{idx}",
                "chunk": c
            }
            out_slide.write(json.dumps(record, ensure_ascii=False) + "\n")
