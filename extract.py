import os
import json
#cambiarlo a pdf plumber
from pypdf import PdfReader
from tqdm import tqdm
from datetime import datetime
from dateutil.parser import parse as date_parse

input_folder = "pdf_remover"
output_file = "output/prueba.jsonl"

def extract_metadata(reader, filename):
    metadata = reader.metadata or {}

    # Fallos seguros para metadata PDF
    author = metadata.get("/Author", "Desconocido")
    creation_date = metadata.get("/CreationDate", None)

    # Intentar parsear fecha
    if creation_date:
        try:
            # Formato PDF: D:YYYYMMDDHHmmSS
            creation_date = creation_date.replace("D:", "")
            creation_date = date_parse(creation_date).isoformat()
        except:
            creation_date = None

    return {
        "autor": author if author else "Desconocido",
        "fecha": creation_date,
        "fuente_archivo": filename,
        "curso": "Inteligencia Artificial — II Semestre 2025"
    }

def extract_text(reader):
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()

os.makedirs("output", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as jsonl:
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(input_folder, filename)
            try:
                reader = PdfReader(filepath)
                text = extract_text(reader)
                metadata = extract_metadata(reader, filename)

                record = {
                    "id": filename,
                    "titulo": os.path.splitext(filename)[0],
                    "contenido": text,
                    "tags": [],
                    **metadata
                }

                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error leyendo {filename}: {e}")