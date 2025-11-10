# websearch_tool.py
from ddgs import DDGS
from langchain_core.tools import StructuredTool  # versión moderna
from pydantic import BaseModel, Field
from urllib.parse import urlparse
import unicodedata
class WebSearchInput(BaseModel):
    query: str = Field(..., description="Consulta del usuario para buscar en la web")
    max_results: int = Field(default=3, description="Máximo número de resultados")

def _clean_query(query: str) -> str:
    """Normaliza la consulta para eliminar acentos y caracteres especiales."""
    return unicodedata.normalize("NFKD", query).encode("ASCII", "ignore").decode()

def _search_duckduckgo(query: str, max_results: int):
    """
    Devuelve un par (contexto, referencias), donde:
    - contexto: string con títulos y snippets de los resultados
    - referencias: lista de dicts con 'documento' (título) y 'autor' (dominio)
    """
    clean_q = _clean_query(query)
    context_lines = []
    referencias = []

    with DDGS() as ddgs:
        for r in ddgs.text(clean_q, max_results=max_results):
            title = r.get("title", "").strip()
            snippet = r.get("body", "").strip()
            href = r.get("href", "")
            domain = urlparse(href).netloc

            context_lines.append(f"{title} — {snippet}")
            referencias.append({
                "documento": title or domain,
                "autor": domain
            })

    # Crear respuesta formateada igual que el RAG Tool
    contexto = "\n".join(context_lines)
    #refs_text = "\n".join([f"- {ref['documento']} — {ref['autor']}" for ref in referencias])

    return contexto, referencias   

# Función expuesta a LangChain — ahora devuelve string formateado completo
def web_search(query: str, max_results=3):
    return _search_duckduckgo(query, max_results)

# Registro del tool (manteniendo compatibilidad con .run)
websearch_tool = StructuredTool(
    name="WebSearch_Tool",
    func=web_search,
    description="Realiza una búsqueda web (DuckDuckGo) y devuelve contexto + referencias formateadas.",
    args_schema=WebSearchInput,
)
