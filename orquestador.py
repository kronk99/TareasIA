import os
import openai
from rag_tool import rag_tool
from websearch_tool import websearch_tool
from rag_tool_sliding import rag_tool_sliding
# Inicializa el cliente
openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_base = "Eres IA-Tutor ,"\
"un asistente académico especializado en apuntes de Inteligencia Artificial (2 semestre 2025)"\
"Hablas con un tono amigable y claro."\
"Tu rol es responder preguntas basadas en los documentos; siempre citas el documento y el autor donde obtienes la información."\
"Usa la RAG tool para extraer respuestas de la base vectorial y solo utiliza la WebSearch tool si el usuario lo solicita explícitamente."\
"Decide qué herramienta usar para responder la pregunta del usuario. "\
"Responde solo con una palabra: 'websearch' si el usuario pidió buscar en internet "\
"No inventes datos ni respondas fuera del dominio."\
"Mantén la coherencia con preguntas anteriores durante la sesión actual." \
"Cuando utilices la RAG Tool, analiza los fragmentos recuperados y genera una " \
"respuesta clara, concisa y precisa . Usa tus capacidades de síntesis para responder " \
"a la pregunta del usuario con tus propias palabras basándote en el contexto entregado." \
" Cita al final los documentos y autores de los fragmentos utilizados"

#orden de secuencias
#1 busca con la pregunta que haga el usuario si se usa el websearch o se usa la base de datos
#obtener el rag / werbsearchtool
#caso del rag:
#se obtiene el rag, se pasa explicitamente la pregunta
#se obtiene la respuesta del rag como par respuesta,fuentes
#se manda el rag y las fuentes al modelo nuevamente para afinar una respuesta, con el prompt base más la pregunta y el historial.
#se da la respuesta

#segmenta la respuesta en un par respuesta, referencias
def parse_rag_output(rag_output: str):
    """
    Separa el texto de los fragmentos y las referencias del string devuelto por rag_tool.run().
    Devuelve (contexto, referencias).
    """
    # Normaliza saltos de línea
    rag_output = rag_output.strip()
    # Divide por "Referencias:"
    if "Referencias:" in rag_output:
        context_part, refs_part = rag_output.split("Referencias:", 1)
    else:
        # En caso de que no haya referencias
        context_part, refs_part = rag_output, ""
    # Elimina la etiqueta "Respuesta:"
    context = context_part.replace("Respuesta:", "").strip()
    # Procesa referencias por líneas
    refs = []
    for line in refs_part.split("\n"):
        line = line.strip("- ").strip()
        if line:
            # Formato esperado: Documento — Autor
            parts = line.split("—")
            doc = parts[0].strip()
            aut = parts[1].strip() if len(parts) > 1 else ""
            refs.append({"documento": doc, "autor": aut})
    return context, refs
#ultimos_msgs = st.session_state.historial[-6:]  # 3 pares usuario-agente = 6 entradas
def decide_and_respond(user_question: str, history: list , type_rag_tool:str):
    #orquestador principal, decide que herramienta utilizar y construye la respuesta final

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
        temperature=0.20  # temperatura baja para respuestas más determinísticas
    )

    assistant_reply = response.choices[0].message.content.strip().lower()

    # Decidir en base a la respuesta del orquestador, esto se debe de cambiar , se asume que buscar en internet incluye buscar
    #en internet +orquestador del rag
    if any(keyword in assistant_reply for keyword in ["buscar en internet", "busca en internet", "websearch", "internet"]):
        print("🔎 Modo WebSearch activado")
    
        # Ejecutar búsqueda web (ahora devuelve un par)
        context, refs = websearch_tool.run(user_question)
        
        # Retornar directamente los resultados crudos sin pasar por GPT
        return context, refs
    
    else:
        print("entre al rag")
        # El modelo decidió usar RAG , pero el usuario define cual
        if (type_rag_tool == "sliding"):
            rag_fragments = rag_tool_sliding.run(user_question)
            context, refs = parse_rag_output(rag_fragments)
            # 4) Crear un nuevo prompt para generar la respuesta a partir del contexto
            messages_summary = [{"role": "system", "content": prompt_base}]
            for h in history:
                role = "user" if h["role"] == "user" else "assistant"
                messages.append({"role": role, "content": h["text"]})
            # Incluir el contexto recuperado
            messages_summary.append({
                "role": "user",
                "content": (
                    f"Contexto recuperado:\n{context}\n\n"
                    f"Ahora, en base a este contexto, responde a la pregunta: {user_question}"
                ),
            })

            # Pedir al modelo que sintetice la respuesta
            summary_response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages_summary,
                max_tokens=200,
                temperature=0.2,
            ).choices[0].message.content.strip()

            return summary_response, refs
        else: #caso modelo B, por saltos de linea
            rag_fragments = rag_tool.run(user_question)
            context, refs = parse_rag_output(rag_fragments)
            # 4) Crear un nuevo prompt para generar la respuesta a partir del contexto
            messages_summary = [{"role": "system", "content": prompt_base}]
            for h in history[-6:]:
                role = "user" if h["role"] == "user" else "assistant"
                messages_summary.append({"role": role, "content": h["text"]})
            # Incluir el contexto recuperado
            messages_summary.append({
                "role": "user",
                "content": (
                    f"Contexto recuperado:\n{context}\n\n"
                    f"Ahora, en base a este contexto, responde a la pregunta: {user_question}"
                ),
            })

            # le pide al modelo una mejor sintesis del rag
            summary_response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages_summary,
                max_tokens=200,
                temperature=0.2,
            ).choices[0].message.content.strip()

            return summary_response, refs
        
    
def construir_respuesta(pregunta: str, respuesta_final: str, fuentes: list):
    """
    Devuelve un string con la respuesta final y referencias, listo para mostrar al usuario.
    """
    salida = respuesta_final
    if fuentes:
        ref_lines = ["\nReferencias:"]
        for f in fuentes:
            ref_lines.append(f"- {f['documento']} — {f['autor']}")
        salida += "\n" + "\n".join(ref_lines)
    # Opcional: incluir la pregunta al inicio
    return f"{pregunta}\n{salida}"

