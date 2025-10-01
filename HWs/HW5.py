import sys
import importlib
import streamlit as st
import chromadb
from openai import OpenAI

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
CHROMA_DB_PATH = "./ChromaDB_for_lab"

def get_openai_client():
    return OpenAI(api_key=OPENAI_KEY)

@st.cache_resource(show_spinner=False)
def get_chromadb_collection():
    chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("Lab4Collection")
    return collection

def get_relevant_course_info(query, collection, openai_client, n=8):
    embedding_resp = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = embedding_resp.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=n)
    docs = []
    if results and results.get("ids", [[]])[0]:
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            doc_id = results['ids'][0][i]
            docs.append((doc_id, doc_text))
    return docs

def get_short_term_history(messages, limit=5):
    return messages[-limit:] if len(messages) > limit else messages

st.set_page_config(page_title="HW5: Short Term Memory Chatbot", layout="wide")
st.title("HW5: Short Term Memory Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

collection = get_chromadb_collection()
openai_client = get_openai_client()

st.sidebar.header("Chatbot Controls")
st.sidebar.markdown("**Conversation memory type:**")
st.sidebar.write("Buffer: 5 questions")

language = st.sidebar.selectbox(
    "Output language",
    ["English", "Español", "Français", "Russian", "Deutsch", "中文", "日本語"],
    index=0,
    key="chat_lang",
)

style = st.sidebar.selectbox(
    "Response style",
    ["Normal", "100 words", "2 connected paragraphs", "5 bullet points"],
    index=0,
    key="chat_style",
)

if st.sidebar.button("Clear chat history", key="clear_history"):
    st.session_state.messages = []

st.header("Ask about SU Organizations!")

# --- SHOW CURRENT DOCUMENTS IN VECTOR DB ---
all_ids = collection.get()["ids"]
if all_ids:
    with st.expander("Current documents in the vector database:"):
        for doc_id in all_ids:
            st.write(f"- {doc_id}")
else:
    st.warning("No club documents in the vector database yet.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your question about the course...")

def style_instruction(style: str) -> str:
    if style == "100 words":
        return "Answer in about 100 words."
    if style == "2 connected paragraphs":
        return "Answer in exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Provide exactly five concise bullet points."
    return ""

def build_rag_prompt(user_query, relevant_docs, language, style):
    if relevant_docs:
        system_prompt = f"""You are a helpful SU organizations assistant. Use the information below, extracted from club documents, to answer the user's question. If the information is found in the club documents, say "Based on the provided club materials..." at the beginning of your response. If not, say "I am answering from general knowledge." Do not make up information not found in the documents.

Club documents:
"""
        for i, (doc_id, doc_text) in enumerate(relevant_docs):
            system_prompt += f"\n--- Document {i+1}: {doc_id} ---\n"
            system_prompt += doc_text[:1500]
        system_prompt += (
            f"\n\nUser question: {user_query}\n"
            f"Write your answer in {language}. {style_instruction(style)}"
        )
    else:
        system_prompt = (
            f"You are a helpful SU organizations assistant. The user asked: '{user_query}'. "
            "You have no club documents to consult, so answer from general knowledge. "
            f"Write your answer in {language}. {style_instruction(style)}"
        )
    return system_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    relevant_docs = get_relevant_course_info(prompt, collection, openai_client, n=8)

    rag_prompt = build_rag_prompt(
        user_query=prompt,
        relevant_docs=relevant_docs,
        language=language,
        style=style
    )

    if relevant_docs:
        with st.expander("Club Documents Used as Context (RAG)", expanded=False):
            for i, (doc_id, doc_text) in enumerate(relevant_docs):
                st.markdown(f"**Document {i+1}: {doc_id}**")
                st.write(doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""))

    trimmed_history = get_short_term_history(st.session_state.messages, 5)
    model_messages = [{"role": "system", "content": rag_prompt}] + trimmed_history

    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=model_messages,
        temperature=1,
    )
    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

if st.button("Clear chat history"):
    st.session_state.messages = []