# websearch_tool.py
from duckduckgo_search import DDGS
from langchain_core.tools import StructuredTool  # versión moderna
from pydantic import BaseModel, Field
class WebSearchInput(BaseModel):
    query: str = Field(..., description="Consulta que el usuario desea buscar en la web")
    max_results: int = Field(default=3, description="Cantidad máxima de resultados a devolver")
def web_search(query: str, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"{r['title']} - {r['href']}")
    return "\n".join(results)
websearch_tool = StructuredTool(
    name="WebSearch_Tool",
    func=web_search,
    description="Realiza una búsqueda en Internet para obtener información actualizada.",
    args_schema=WebSearchInput,
)
