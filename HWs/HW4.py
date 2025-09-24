import streamlit as st
import os
import sys
import importlib
from io import BytesIO

# Patch sqlite for chromadb compatibility (for Streamlit Cloud etc.)
try:
    sys.modules['sqlite3'] = importlib.import_module('pysqlite3')
except ModuleNotFoundError:
    pass

import chromadb
from openai import OpenAI
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

# ========== API KEYS ==========
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY", "")
COHERE_KEY = st.secrets.get("COHERE_API_KEY", "")

if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is not set! Please configure your key in Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="HW 4: Course Info Chatbot", layout="wide")
st.title("HW 4: Course Information Chatbot (with RAG and HTML)")

# ========== ChromaDB SETUP ==========
CHROMA_DB_PATH = "./ChromaDB_for_lab"

def get_openai_client():
    return OpenAI(api_key=OPENAI_KEY)

@st.cache_resource(show_spinner=False)
def get_chromadb_collection():
    chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("Lab4Collection")
    return collection

# ========== HTML chunking and vector DB creation ==========

def fixed_size_chunking(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_text(text):
    return fixed_size_chunking(text, chunk_size=500, overlap=100)

def is_informative_chunk(chunk):
    ignore_phrases = [
        "requires JavaScript",
        "'Cuse Activities",
        "No Response",
        "About Contact Information"
    ]
    return not any(phrase in chunk for phrase in ignore_phrases)

def extract_html_text(filepath):
    """Extracts visible text from HTML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n")

def create_vector_db_if_needed():
    """Creates the vector database from HTML docs if not already present."""
    if not os.path.exists(CHROMA_DB_PATH):
        # Find su_orgs folder relative to this file
        html_dir = os.path.join(os.path.dirname(__file__), "..", "su_orgs")
        html_dir = os.path.abspath(html_dir)

        if not os.path.exists(html_dir):
            st.error(f"Directory '{html_dir}' does not exist. Please check your project structure.")
            st.stop()

        html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
        chunks = []
        for fname in html_files:
            text = extract_html_text(os.path.join(html_dir, fname))
            doc_chunks = chunk_text(text)
            for i, chunk in enumerate(doc_chunks):
                if not is_informative_chunk(chunk): continue
                chunk_id = f"{fname}_chunk{i+1}"
                chunks.append((chunk_id, chunk))
        chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection("Lab4Collection")
        openai_client = get_openai_client()
        for chunk_id, chunk_text_val in chunks:
            resp = openai_client.embeddings.create(input=chunk_text_val, model="text-embedding-3-small")
            embedding = resp.data[0].embedding
            collection.add(
                documents=[chunk_text_val],
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"filename": chunk_id}]
            )
        print("Vector DB created and filled!")
    else:
        print("Vector DB already exists, not recreating.")

# Only run on first load
create_vector_db_if_needed()

# ========== PDF upload (optional, can be removed if only HTML needed) ==========
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts).strip()

def add_pdf_to_collection(collection, openai_client, file_bytes, filename):
    text = extract_pdf_text(file_bytes)
    if not text.strip():
        return False
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding],
        metadatas=[{"filename": filename}]
    )
    return True

# --- Singleton clients in session_state ---
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = get_openai_client()
if 'Lab4_vectorDB' not in st.session_state:
    st.session_state.Lab4_vectorDB = get_chromadb_collection()
collection = st.session_state.Lab4_vectorDB
openai_client = st.session_state.openai_client

# ========== UPLOAD PDF SECTION ==========
st.header("1. Upload your course PDFs to the vector database (optional)")
uploaded_files = st.file_uploader(
    "Upload one or more course PDF files (syllabus, lectures, etc.)",
    type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing and embedding uploaded files...")
    added = 0
    for up_file in uploaded_files:
        ids = collection.get(ids=[up_file.name])["ids"]
        if up_file.name not in ids:
            if add_pdf_to_collection(collection, openai_client, up_file.read(), up_file.name):
                st.success(f"Added: {up_file.name}")
                added += 1
            else:
                st.warning(f"Could not extract text from: {up_file.name}")
        else:
            st.info(f"Already in DB: {up_file.name}")
    if added == 0:
        st.info("No new PDFs were added.")

all_ids = collection.get()["ids"]
if all_ids:
    with st.expander("Current documents in the vector database:"):
        for doc_id in all_ids:
            st.write(f"- {doc_id}")
else:
    st.warning("No documents in the vector database yet.")

if st.button("🗑️ Clear the vector database (remove all PDFs and HTML chunks from memory)"):
    try:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            st.success("All files have been removed from the vector database.")
        else:
            st.info("The collection is already empty.")
    except Exception as e:
        st.error(f"Error while deleting: {e}")

st.markdown("---")
st.info("You can upload more PDFs at any time. The ChromaDB collection persists between runs (as long as the file system is kept).")

# ========== CHATBOT SECTION with RAG ==========
def style_instruction(style: str) -> str:
    if style == "100 words":
        return "Answer in about 100 words."
    if style == "2 connected paragraphs":
        return "Answer in exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Provide exactly five concise bullet points."
    return ""

def trim_messages_last_n_pairs(messages, n_pairs=5):
    if n_pairs <= 0:
        return []
    tail = []
    user_count = 0
    for m in reversed(messages):
        tail.append(m)
        if m.get("role") == "user":
            user_count += 1
            if user_count >= n_pairs:
                break
    return list(reversed(tail))

def fetch_relevant_docs(query, collection, openai_client, n=8):
    if not query or not collection:
        return []
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

def build_rag_prompt(user_query, relevant_docs, language, style):
    if relevant_docs:
        system_prompt = f"""You are a helpful course information assistant. Use the information below, extracted from course documents, to answer the user's question. If the information is found in the course documents, say "Based on the provided course materials..." at the beginning of your response. If not, say "I am answering from general knowledge." Do not make up information not found in the documents.

Course documents:
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
            f"You are a helpful course information assistant. The user asked: '{user_query}'. "
            "You have no course documents to consult, so answer from general knowledge. "
            f"Write your answer in {language}. {style_instruction(style)}"
        )
    return system_prompt

# ========== SIDEBAR ==========
st.sidebar.header("Chatbot Controls")
llm_vendor = st.sidebar.selectbox(
    "Choose LLM Vendor",
    [
        "OpenAI",
        "Mistral",
        "Cohere",
    ],
    index=0,
)
if llm_vendor == "OpenAI":
    llm_model = st.sidebar.selectbox(
        "Select OpenAI model",
        [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-5-nano"
        ],
        key="openai_model",
        index=0
    )
elif llm_vendor == "Mistral":
    llm_model = st.sidebar.selectbox(
        "Select Mistral model",
        [
            "mistral-large-latest",
            "mistral-medium",
            "mistral-small",
        ],
        key="mistral_model",
        index=0
    )
elif llm_vendor == "Cohere":
    llm_model = st.sidebar.selectbox(
        "Select Cohere model",
        [
            "command-a-03-2025",
            "command-r-08-2024",
        ],
        key="cohere_model",
        index=0
    )
else:
    llm_model = "gpt-4o"

memory_type = st.sidebar.selectbox(
    "Conversation memory type",
    [
        "Buffer: 5 questions",
        "Conversation summary",
        "Buffer: 2000 tokens"
    ],
    index=0,
)

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
    st.session_state.awaiting_more_info = False

# ========== MAIN CHAT ==========
st.header("2. Course Information Chatbot (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me any question about the course..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG: Retrieve top-8 relevant course docs for the user query
    relevant_docs = fetch_relevant_docs(prompt, collection, openai_client, n=8)

    # Compose system prompt for LLM
    rag_prompt = build_rag_prompt(
        user_query=prompt,
        relevant_docs=relevant_docs,
        language=language,
        style=style
    )

    # Optionally show which documents are used as context
    if relevant_docs:
        with st.expander("Course Documents Used as Context (RAG)", expanded=False):
            for i, (doc_id, doc_text) in enumerate(relevant_docs):
                st.markdown(f"**Document {i+1}: {doc_id}**")
                st.write(doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""))

    # Conversation memory
    if memory_type == "Buffer: 6 questions":
        trimmed_history = trim_messages_last_n_pairs(st.session_state.messages, 6)
    else:
        trimmed_history = st.session_state.messages[-6:]

    model_messages = [{"role": "system", "content": rag_prompt}] + trimmed_history

    def stream_openai_response(model, messages, api_key):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=1,
            )
            return st.write_stream(stream)
        except Exception as e:
            st.error(f"OpenAI streaming error: {e}")
            return ""

    def stream_mistral_response(model, messages, api_key):
        import json
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream"
        }
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }
        try:
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
            if not response.ok:
                st.error(f"Mistral HTTP error: {response.status_code} {response.text}")
                return ""
            buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    content = line.removeprefix("data: ").strip()
                    if content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            text = delta["content"]
                            buffer += text
                            yield text
                    except Exception:
                        continue
            yield ""
        except Exception as e:
            st.error(f"Mistral streaming error: {e}")
            return ""

    def stream_cohere_response(model, messages, api_key):
        import json
        url = "https://api.cohere.ai/v1/chat"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        def cohere_role(role):
            return "USER" if role == "user" else "CHATBOT"
        data = {
            "model": model,
            "message": messages[-1]["content"] if messages else "",
            "chat_history": [
                {"role": cohere_role(m["role"]), "message": m["content"]}
                for m in messages[:-1] if m["role"] in ["user", "assistant"]
            ],
            "stream": True,
            "temperature": 0.7,
        }
        try:
            with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as resp:
                if not resp.ok:
                    st.error(f"Cohere HTTP error: {resp.status_code} {resp.text}")
                    return ""
                buffer = ""
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "event_type" in chunk and chunk["event_type"] == "text-generation":
                                text = chunk.get("text", "")
                                buffer += text
                                yield text
                            if "event_type" in chunk and chunk["event_type"] == "stream-end":
                                break
                        except Exception:
                            continue
                yield ""
        except Exception as e:
            st.error(f"Cohere streaming error: {e}")
            return ""

    # LLM response
    if llm_vendor == "OpenAI":
        if not OPENAI_KEY:
            with st.chat_message("assistant"):
                st.error("OpenAI API key not found.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = stream_openai_response(llm_model, model_messages, OPENAI_KEY)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    elif llm_vendor == "Mistral":
        if not MISTRAL_KEY:
            with st.chat_message("assistant"):
                st.error("Mistral API key not found.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = st.write_stream(
                        stream_mistral_response(llm_model, model_messages, MISTRAL_KEY)
                    )
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    elif llm_vendor == "Cohere":
        if not COHERE_KEY:
            with st.chat_message("assistant"):
                st.error("Cohere API key not found.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = st.write_stream(
                        stream_cohere_response(llm_model, model_messages, COHERE_KEY)
                    )
            st.session_state.messages.append({"role": "assistant", "content": response_text})