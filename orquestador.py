import os
import openai
from rag_tool import rag_tool
from websearch_tool import websearch_tool
# Inicializa el cliente
openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_base = "Eres IA-Tutor ,"\
"un asistente académico especializado en apuntes de Inteligencia Artificial (2 semestre 2025)"\
"Hablas con un tono amigable y claro."\
"Tu rol es responder preguntas basadas en los documentos; siempre citas el documento y el autor donde obtienes la información."\
"Usa la RAG tool para extraer respuestas de la base vectorial y solo utiliza la WebSearch tool si el usuario lo solicita explícitamente."\
"No inventes datos ni respondas fuera del dominio."\
"Mantén la coherencia con preguntas anteriores durante la sesión actual."

#orden de secuencias
#1 busca con la pregunta que haga el usuario si se usa el websearch o se usa la base de datos
#obtener el rag / werbsearchtool
#caso del rag:
#se obtiene el rag, se pasa explicitamente la pregunta
#se obtiene la respuesta del rag como par respuesta,fuentes
#se manda el rag y las fuentes al modelo nuevamente para afinar una respuesta, con el prompt base más la pregunta y el historial.
#se da la respuesta

#ultimos_msgs = st.session_state.historial[-6:]  # 3 pares usuario-agente = 6 entradas
def decide_and_respond(user_question: str, history: list):
    # structura del mensaje para chatcompletition
    messages = [{"role": "system", "content": prompt_base}]

    #se añade el rol y el texto en el historial , revisar si añade correctamente el par pregunta y respuesta
    #se debe de limitar a 6 elementos 3 preguntas usuario 3 respuestas agente
    for h in history:
        role = "user" if h["role"] == "user" else "assistant"
        messages.append({"role": role, "content": h["text"]})

    # Añade la pregunta actual como último mensaje de usuario
    messages.append({"role": "user", "content": user_question})

    # Invoca al modelo para decidir qué herramienta usar
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=50,
        temperature=0.0  # temperatura baja para respuestas más determinísticas
    )

    assistant_reply = response.choices[0].message.content.strip().lower()

    # Decidir en base a la respuesta del orquestador, esto se debe de cambiar , se asume que buscar en internet incluye buscar
    #en internet +orquestador del rag
    if "buscar en internet" in assistant_reply or "websearch" in assistant_reply:
        # El modelo decidió que necesita una búsqueda web
        web_result = websearch_tool.run(user_question)
        return web_result, []  # revisar si retorna un par respuesta , fuentes
    else:
        # El modelo decidió usar RAG
        rag_fragments = rag_tool.run(user_question)
        #CONSUltar si esto retorna par fragmento, fuentes
        return rag_fragments, []  # incluiríamos metadatos de fuentes
    
def construir_respuesta(pregunta: str, frag_textos: str, fuentes: list, rag: str):
    respuesta = f"Respuesta generada para: {pregunta} usando {rag}\n\n{frag_textos}"
    # Añade secciones de referencias ARREGLAR ESTO DEL ORQUESTADOR, EL PAR ES RESPUESTA, FUENTES
    if fuentes:
        ref_lines = ["\nReferencias:"]
        for f in fuentes:
            ref_lines.append(f"- {f['titulo']} — {f.get('autor','')}")
        respuesta += "\n".join(ref_lines)
    return respuesta,fuentes 

# Ejemplo de uso:
frag_textos, fuentes = decide_and_respond("¿Qué es un agente inteligente?", history=[])
print(construir_respuesta("¿Qué es un agente inteligente?", frag_textos, fuentes, rag="RAG Tool"))