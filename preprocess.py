import json
import unicodedata
import re
import os

INPUT_DIR  = "clean_texts"         # nuevo archivo con el formato proporcionado
OUT_PARAGRAPH = "output/documents_parrafos.jsonl"

def normalize_text(text: str) -> str:
    """
    Normaliza texto a minúsculas y elimina caracteres no deseados,
    pero conserva los saltos de línea para permitir la segmentación en párrafos.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    # Unificar \r\n y \r en \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Sustituir múltiples espacios o tabulaciones por un solo espacio (sin tocar \n)
    text = re.sub(r"[ \t]+", " ", text)
    # Eliminar cualquier carácter no alfanumérico, acentos, puntuación básica y saltos de línea
    text = re.sub(r"[^\x00-\x7Fñáéíóúü°%()¡!¿?.,;:0-9a-z\n ]", "", text)
    return text.strip()

def split_paragraphs(text: str):
    """Divide el texto en párrafos por cada salto de línea."""
    return [ln.strip() for ln in text.split("\n") if len(ln.strip().split()) > 3]

# Crear carpeta de salida si no existe
os.makedirs(os.path.dirname(OUT_PARAGRAPH), exist_ok=True)

# Abrir archivo de salida una sola vez
with open(OUT_PARAGRAPH, "w", encoding="utf-8") as out_par:

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
        base_name = "_".join(parts[:4])
        sub_id = parts[4].split("-")[0]
        chunk_base_id = f"{base_name}_{sub_id}.pdf"

        documento = base_name
        autor = doc["metadata"].get("author", "Desconocido")

        parag_index = 0
        for page in doc.get("pages", []):
            raw_text = page.get("text", "")
            norm_text = normalize_text(raw_text)
            paragraphs = split_paragraphs(norm_text)

            for p in paragraphs:
                record = {
                    "chunk_id": f"{chunk_base_id}_p{parag_index}",
                    "chunk": p,
                    "autor": autor,
                    "documento": documento,
                }
                out_par.write(json.dumps(record, ensure_ascii=False) + "\n")
                parag_index += 1

        print(f"✅ Procesado: {file} → {parag_index} párrafos extraídos.")
