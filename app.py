import streamlit as st
# app.py
import streamlit as st
import orquestador
#cosas por mejorar.--------------------
#1 definir roles del agente , no solo agent
#-------------------
# Se supone una funcion que recibe la pregunta 
# y devuelve (texto_respuesta, lista_de_fuentes)
def call_agent(pregunta: str, rag: str, busqueda_web: bool): #agregar lo de busqueda web
    "este es el metood para llamar al agente"
    "rag identifica la estrategia de segmentacion que se escoge"
    "busqueda web es un checkbox para usar websearchtool"
    "historial son los mensajes anteriores, devuelve un par respuesta,fuentes"
    # Estilo de formato de ejemplo a seguir
    #aca se debe de hacer las llamadas para las apis y todo el toolchain del chatbot

    #respuesta = f"Respuesta generada para: {pregunta} usando {rag}"
    #fuentes = [
        #{"titulo": "Apunte1.pdf", "autor": "Estudiante A"},
        #{"titulo": "Apunte2.pdf", "autor": "Estudiante B"}
    #]
    #return respuesta, fuentes
    ultimos_msgs = st.session_state.historial[-6:]
    frag_textos,fuentes = orquestador.decide_and_respond(pregunta,ultimos_msgs)
    respuesta = orquestador.construir_respuesta(pregunta ,frag_textos,fuentes,rag)
    return respuesta

# Configuración inicial de Streamlit
st.set_page_config(page_title="Chat RAG", page_icon="💬", layout="wide")
st.title(" Agente Conversacional para Apuntes")

# Inicializar estados de sesión
if "historial" not in st.session_state:
    st.session_state.historial = []        # lista de mensajes de la conversación actual
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []    # lista de conversaciones previas en la sesión
if "agente_nombre" not in st.session_state:
    st.session_state.agente_nombre = "Agente IA"
if "rag" not in st.session_state:
    st.session_state.rag = "Segmentación A"
if "busqueda_web" not in st.session_state:
    st.session_state.busqueda_web = False
with st.sidebar:
    # Centrar el botón "Nuevo chat"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Nuevo chat"):
            # Guardar el chat actual en el historial de sesiones si no está vacío
            if st.session_state.historial:
                st.session_state.chat_sessions.append(
                    st.session_state.historial.copy()
                )
            # Reiniciar la conversación actual
            st.session_state.historial = []

    st.markdown("---")
    st.markdown("### Historial de chats (sesión actual)")
    # Mostrar cada chat guardado (solo mientras dure la sesión)
    if st.session_state.chat_sessions:
        for idx, chat in enumerate(reversed(st.session_state.chat_sessions), 1):
            nombre_chat = f"Chat {len(st.session_state.chat_sessions)-idx+1}"
            st.markdown(f"- {nombre_chat}")
    else:
        st.write("No hay chats guardados en esta sesión")

# Mostrar mensajes previos de la conversación actual
for mensaje in st.session_state.historial:
    emisor = "user" if mensaje["role"] == "user" else "assistant"
    nombre = "Tú" if mensaje["role"] == "user" else st.session_state.agente_nombre
    with st.chat_message(emisor):
        st.markdown(f"**{nombre}:** {mensaje['text']}")
        # Mostrar fuentes (si las hay)
        if mensaje.get("sources"):
            for src in mensaje["sources"]:
                st.markdown(f"🔖 *Fuente:* {src['titulo']} — {src.get('autor', '')}")

# Campo de texto para preguntar, aca se debe de tomar para la el llamado
#a la funcion del agente
pregunta_usuario = st.text_input(
    "Escribe tu pregunta…", key="user_input"
)

# Opciones de configuración debajo del cuadro de texto
st.markdown("#### Configuración de la consulta")
st.session_state.rag = st.selectbox(
    "Seleccionar RAG (segmentación)",
    ["Segmentación A", "Segmentación B"],
    index=["Segmentación A", "Segmentación B"].index(st.session_state.rag),
)
st.session_state.busqueda_web = st.checkbox(
    "Permitir búsquedas web",
    value=st.session_state.busqueda_web
)

# Botón enviar 
if st.button("Enviar"):
    if pregunta_usuario.strip():
        # Guardar pregunta en la conversación actual
        st.session_state.historial.append({
            "role": "user",
            "text": pregunta_usuario.strip(),
        })

        # Llamar al agente con la configuración seleccionada
        with st.spinner("Escribiendo…"):
            
            #cambiar esto por el nombre correcto para la funcion 
            respuesta, fuentes = call_agent(
                pregunta_usuario,
                st.session_state.rag,
                st.session_state.busqueda_web,
                historial_textos,
            )

        # Guardar respuesta del agente
        st.session_state.historial.append({
            "role": "agent",
            "text": respuesta,
            "sources": fuentes,
        })

        # Limpiar el campo de texto para la siguiente pregunta
        st.session_state.user_input = ""

        # Refrescar la página para mostrar la nueva conversación
        st.experimental_rerun()