import json
import unicodedata
import re
import os

INPUT_DIR      = "clean_texts"           # Carpeta con tus JSON de apuntes
OUT_SLIDING    = "output/documents_sliding.jsonl"

CHUNK_SIZE = 120   # palabras por chunk para sliding window
OVERLAP    = 40   # solapamiento

def normalize_text(text: str) -> str:
    """Normaliza texto a minúsculas y elimina caracteres no deseados,
    pero conserva los saltos de línea para la segmentación."""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x00-\x7Fñáéíóúü°%()¡!¿?.,;:0-9a-z\n ]", "", text)
    return text.strip()

def split_paragraphs(text: str):
    """Divide el texto en párrafos por cada salto de línea y descarta líneas muy cortas."""
    return [ln.strip() for ln in text.split("\n") if len(ln.strip().split()) > 3]

def sliding_window(words, size, overlap):
    """Genera ventanas deslizantes de 'size' palabras con solapamiento."""
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+size]
        # Solo guardar si tiene suficientes palabras
        if len(chunk) < 30:
            break
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks


with open(OUT_SLIDING,  "w", encoding="utf-8") as out_slide:

    # Iterar sobre todos los archivos JSON de la carpeta
    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".json"):
            continue

        input_path = os.path.join(INPUT_DIR, file)

        try:
            with open(input_path, "r", encoding="utf-8") as infile:
                doc = json.load(infile)
        except Exception as e:
            print(f"⚠️ Error al leer {file}: {e}")
            continue

        file_name = doc["file_name"]
        parts = file_name.split("_", 4)
        base_name = "_".join(parts[:4])     # ejemplo: "10_SEMANA_AI_20251007"
        sub_id = parts[4].split("-")[0]     # ejemplo: "1"
        chunk_base_id = f"{base_name}_{sub_id}.pdf"

        documento = base_name
        autor = doc["metadata"].get("author", "Desconocido")

        slide_index  = 0  # Contador para sliding windows

        for page in doc.get("pages", []):
            raw_text = page.get("text", "")
            norm_text = normalize_text(raw_text)
            # --- Segmentación por ventana deslizante ---
            words = norm_text.split()
            chunks = sliding_window(words, CHUNK_SIZE, OVERLAP)
            for c in chunks:
                record = {
                    "chunk_id": f"{chunk_base_id}_s{slide_index}",
                    "chunk": c,
                    "autor": autor,
                    "documento": documento,
                }
                out_slide.write(json.dumps(record, ensure_ascii=False) + "\n")
                slide_index += 1

        print(f"✅ Procesado: {file} → {slide_index} ventanas")
