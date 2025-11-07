from langchain.tools import Tool
from duckduckgo_search import DDGS

def web_search(query: str, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"{r['title']} - {r['href']}")
    return "\n".join(results)

websearch_tool = Tool(
    name="WebSearch_Tool",
    func=web_search,
    description="Realiza una búsqueda en Internet para obtener información actualizada."
)