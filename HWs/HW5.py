import streamlit as st
import os
import sys
import importlib
from io import BytesIO
import json
from typing import List, Tuple, Dict, Any

# SQLite shim (for some environments where chroma needs pysqlite3)
try:
    sys.modules['sqlite3'] = importlib.import_module('pysqlite3')
except ModuleNotFoundError:
    pass

import chromadb
from openai import OpenAI
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

# ==============================
# KEYS / CONFIG
# ==============================
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY", "")
COHERE_KEY = st.secrets.get("COHERE_API_KEY", "")

if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is not set! Configure it in Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="HW5: Enhanced iSchool Chatbot (Function Calling)", layout="wide")
st.title("HW5: Enhanced iSchool Chatbot (Function Calling + Context Memory)")

CHROMA_DB_PATH = "./ChromaDB_for_hw5"
HTML_SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "su_orgs"))
COLLECTION_NAME = "HW5Collection"

# ==============================
# CLIENTS
# ==============================
def get_openai_client():
    return OpenAI(api_key=OPENAI_KEY)

@st.cache_resource(show_spinner=False)
def get_chromadb_collection():
    chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
    return chroma_client.get_or_create_collection(COLLECTION_NAME)

# ==============================
# UTILITIES
# ==============================
def fixed_size_chunking(text: str, chunk_size=50, overlap=10) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_html_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts).strip()

# ==============================
# VECTOR DB INITIALIZATION (HTML once)
# ==============================
def create_vector_db_if_needed():
    """
    If the DB directory does not exist, build it from HTML sources once.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        if not os.path.exists(HTML_SOURCE_DIR):
            st.warning(f"HTML directory not found: {HTML_SOURCE_DIR}")
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            return
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        st.info("Building initial vector DB from HTML sources...")
        html_files = [f for f in os.listdir(HTML_SOURCE_DIR) if f.endswith(".html")]
        chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        openai_client = get_openai_client()

        for fname in html_files:
            full_path = os.path.join(HTML_SOURCE_DIR, fname)
            text = extract_html_text(full_path)
            if not text.strip():
                continue
            doc_chunks = fixed_size_chunking(text, chunk_size=500, overlap=100)
            for i, chunk_text_val in enumerate(doc_chunks):
                chunk_id = f"{fname}_chunk{i+1}"
                resp = openai_client.embeddings.create(
                    input=chunk_text_val,
                    model="text-embedding-3-small"
                )
                embedding = resp.data[0].embedding
                collection.add(
                    documents=[chunk_text_val],
                    ids=[chunk_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "filename": fname,
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks)
                    }]
                )
        st.success("Initial HTML indexing complete.")

create_vector_db_if_needed()

# ==============================
# ADD PDF UPLOAD
# ==============================
def add_pdf_to_collection(collection, openai_client, file_bytes, filename):
    text = extract_pdf_text(file_bytes)
    if not text.strip():
        return False
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding],
        metadatas=[{"filename": filename, "chunk_index": 0, "total_chunks": 1}]
    )
    return True

# ==============================
# RETRIEVAL
# ==============================
def fetch_relevant_docs(query: str, collection, openai_client, n=5) -> List[Tuple[str, str, Dict]]:
    if not query:
        return []
    emb = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[emb],
        n_results=n
    )
    docs = []
    if results and results.get("ids", [[]])[0]:
        for i in range(len(results["documents"][0])):
            doc_text = results["documents"][0][i]
            doc_id = results["ids"][0][i]
            meta = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {}
            docs.append((doc_id, doc_text, meta))
    return docs

# ==============================
# STYLE HELPERS
# ==============================
def style_instruction(style: str) -> str:
    if style == "100 words":
        return "Answer in about 100 words."
    if style == "2 connected paragraphs":
        return "Answer in exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Provide exactly five concise bullet points."
    return ""

def build_rag_prompt(user_query: str,
                     relevant_docs: List[Tuple[str, str, Dict]],
                     language: str,
                     style: str) -> str:
    if relevant_docs:
        system_prompt = (
            "You are a helpful iSchool assistant. Use ONLY the provided document chunks to answer. "
            "If they contain the answer, begin with 'Based on the provided course materials...' "
            "Otherwise begin with 'I am answering from general knowledge.' "
            "Do not hallucinate. Cite filenames when you use content.\n\n"
            "Document chunks:\n"
        )
        for i, (doc_id, doc_text, meta) in enumerate(relevant_docs):
            fname = meta.get("filename", "unknown")
            system_prompt += f"\n--- Chunk {i+1} (file={fname}, id={doc_id}) ---\n"
            system_prompt += doc_text[:1500]
        system_prompt += (
            f"\n\nUser question: {user_query}\n"
            f"Language: {language}. {style_instruction(style)}"
        )
    else:
        system_prompt = (
            f"You are a helpful iSchool assistant. No documents matched. "
            f"Answer generally. User question: {user_query}\n"
            f"Language: {language}. {style_instruction(style)}"
        )
    return system_prompt

def trim_messages_last_n_pairs(messages: List[Dict], n_pairs=5):
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

# ==============================
# FUNCTION CALLING TOOLS (OpenAI)
# ==============================
def tool_fetch_relevant(query: str) -> str:
    docs = fetch_relevant_docs(query, collection, openai_client, n=5)
    st.session_state.last_retrieved_docs = docs
    if not docs:
        return "NO_RESULTS"
    out_lines = []
    for i, (doc_id, doc_text, meta) in enumerate(docs, start=1):
        fname = meta.get("filename", "unknown")
        snippet = doc_text[:600]
        out_lines.append(f"Result {i}: id={doc_id}, file={fname}, snippet={snippet}")
    return "\n".join(out_lines)

def tool_get_doc_details(index: int) -> str:
    docs = st.session_state.get("last_retrieved_docs", [])
    if not docs:
        return "NO_CONTEXT: no previous retrieval to select from."
    if index < 1 or index > len(docs):
        return f"INVALID_INDEX: only {len(docs)} docs available."
    doc_id, text, meta = docs[index - 1]
    fname = meta.get("filename", "unknown")
    return (
        f"FULL_DOC index={index}, id={doc_id}, file={fname}\n"
        f"CONTENT_START\n{text}\nCONTENT_END"
    )

def call_llm_with_tools(user_input: str,
                        language: str,
                        style: str,
                        model: str) -> str:
    client = openai_client

    system_instruction = (
        "You are an iSchool assistant with function calling.\n"
        "Rules:\n"
        "1) Call fetch_relevant_docs for a genuinely new informational query or when existing retrieved chunks are insufficient.\n"
        "2) If the follow-up clearly refers to the same topic and can be answered using the previously retrieved chunks, answer directly without calling any tool.\n"
        "3) Call get_doc_details when the user wants more detail about a specific previously shown document (e.g. 'second document').\n"
        "4) If the user requests detail with no previous retrieval, ask them to pose an informational query first.\n"
        "5) If a document index is invalid, ask the user to pick a valid one.\n"
        f"Always answer in {language} and respect style instructions: {style_instruction(style)}.\n"
        "Never invent content outside tool outputs."
    )

    history = st.session_state.messages[-12:]

    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "fetch_relevant_docs",
                    "description": "Retrieve relevant document snippets for a new question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doc_details",
                    "description": "Get full content of one of the last retrieved documents by its 1-based index.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "minimum": 1}
                        },
                        "required": ["index"]
                    }
                }
            }
        ],
        tool_choice="auto",
        temperature=0.7
    )

    first_msg = response.choices[0].message

    if not getattr(first_msg, "tool_calls", None):
        return first_msg.content or "(empty response)"

    tool_call = first_msg.tool_calls[0]
    fname = tool_call.function.name
    try:
        fargs = json.loads(tool_call.function.arguments)
    except Exception:
        fargs = {}

    if fname == "fetch_relevant_docs":
        tool_output = tool_fetch_relevant(fargs.get("query", ""))
    elif fname == "get_doc_details":
        tool_output = tool_get_doc_details(fargs.get("index", 0))
    else:
        tool_output = "UNKNOWN_TOOL"

    follow_messages = [
        {
            "role": "system",
            "content": "Use ONLY tool output. If NO_RESULTS -> say no documents found; if INVALID_INDEX or NO_CONTEXT -> ask user to clarify."
        },
        *messages,
        first_msg,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_output
        }
    ]

    follow = client.chat.completions.create(
        model=model,
        messages=follow_messages,
        temperature=0.7
    )
    return follow.choices[0].message.content or "(empty)"

# ==============================
# SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("Chatbot Controls")

llm_vendor = st.sidebar.selectbox(
    "Choose LLM Vendor",
    ["OpenAI", "Mistral", "Cohere"],
    index=0
)

if llm_vendor == "OpenAI":
    llm_model = st.sidebar.selectbox(
        "OpenAI model",
        ["gpt-4o", "gpt-4-turbo", "gpt-5-nano"],
        index=0
    )
elif llm_vendor == "Mistral":
    llm_model = st.sidebar.selectbox(
        "Mistral model",
        ["mistral-large-latest", "mistral-medium", "mistral-small"],
        index=0
    )
else:
    llm_model = st.sidebar.selectbox(
        "Cohere model",
        ["command-r-08-2024", "command-a-03-2025"],
        index=0
    )

memory_type = st.sidebar.selectbox(
    "Conversation memory type",
    ["Buffer: 5 questions", "Conversation summary", "Buffer: 2000 tokens"],
    index=0
)

language = st.sidebar.selectbox(
    "Output language",
    ["English", "Spanish", "French", "German", "Chinese", "Japanese"],
    index=0
)

style = st.sidebar.selectbox(
    "Response style",
    ["Normal", "100 words", "2 connected paragraphs", "5 bullet points"],
    index=0
)

if st.sidebar.button("Clear chat history"):
    st.session_state.messages = []
    st.session_state.last_retrieved_docs = []

# ==============================
# PDF UPLOAD
# ==============================
st.header("1. Upload PDFs (optional)")
openai_client = get_openai_client()
collection = get_chromadb_collection()

uploaded_files = st.file_uploader("Upload course PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    st.info("Processing and embedding uploaded PDF files...")
    added = 0
    for up_file in uploaded_files:
        ids_existing = collection.get(ids=[up_file.name])["ids"]
        if up_file.name not in ids_existing:
            if add_pdf_to_collection(collection, openai_client, up_file.read(), up_file.name):
                st.success(f"Added: {up_file.name}")
                added += 1
            else:
                st.warning(f"Could not extract text from: {up_file.name}")
        else:
            st.info(f"Already in DB: {up_file.name}")
    if added == 0:
        st.info("No new PDFs were added.")

# ==============================
# VECTOR DB LIST
# ==============================
all_ids = collection.get()["ids"]
if all_ids:
    with st.expander("Current documents in vector database"):
        for doc_id in all_ids[:500]:
            st.write(f"- {doc_id}")
        if len(all_ids) > 500:
            st.write(f"... and {len(all_ids)-500} more")
else:
    st.warning("No documents in the vector database yet.")

if st.button("🗑️ Clear the vector database"):
    try:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            st.success("All files removed.")
            st.session_state.last_retrieved_docs = []
        else:
            st.info("The collection is already empty.")
    except Exception as e:
        st.error(f"Error while deleting: {e}")

st.markdown("---")

# ==============================
# CHAT HISTORY DISPLAY
# ==============================
st.header("2. Chat (with Function Calling for OpenAI)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# Fallback streaming (non-OpenAI function calling)
# ==============================
def stream_openai_simple(model, messages, api_key):
    try:
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        return st.write_stream(stream)
    except Exception as e:
        st.error(f"OpenAI streaming error: {e}")
        return ""

def stream_mistral_response(model, messages, api_key):
    import json
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream"}
    data = {"model": model, "messages": messages, "stream": True, "temperature": 0.7}
    try:
        resp = requests.post(url, headers=headers, json=data, stream=True, timeout=90)
        if not resp.ok:
            st.error(f"Mistral HTTP error: {resp.status_code} {resp.text}")
            return ""
        buffer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                payload = line.removeprefix("data: ").strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
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
    def co_role(r):
        return "USER" if r == "user" else "CHATBOT"
    data = {
        "model": model,
        "message": messages[-1]["content"] if messages else "",
        "chat_history": [
            {"role": co_role(m["role"]), "message": m["content"]}
            for m in messages[:-1] if m["role"] in ("user", "assistant")
        ],
        "stream": True,
        "temperature": 0.7
    }
    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=90) as resp:
            if not resp.ok:
                st.error(f"Cohere HTTP error: {resp.status_code} {resp.text}")
                return ""
            buffer = ""
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk = json.loads(line)
                        if chunk.get("event_type") == "text-generation":
                            text = chunk.get("text", "")
                            buffer += text
                            yield text
                        if chunk.get("event_type") == "stream-end":
                            break
                    except Exception:
                        continue
            yield ""
    except Exception as e:
        st.error(f"Cohere streaming error: {e}")
        return ""

# ==============================
# CHAT INPUT
# ==============================
user_prompt = st.chat_input("Ask about organizations or refer to '2nd document', etc...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if llm_vendor == "OpenAI":
        with st.chat_message("assistant"):
            with st.spinner("Thinking (function calling)..."):
                answer = call_llm_with_tools(
                    user_input=user_prompt,
                    language=language,
                    style=style,
                    model=llm_model
                )
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        docs = st.session_state.get("last_retrieved_docs", [])
        if docs:
            with st.expander("Last retrieved documents (numbering reference)"):
                for i, (doc_id, doc_text, meta) in enumerate(docs, start=1):
                    fname = meta.get("filename", "unknown")
                    st.markdown(f"**{i}. {fname} (id={doc_id})**")
                    st.write(doc_text[:800] + ("..." if len(doc_text) > 800 else ""))

    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                retrieved = fetch_relevant_docs(user_prompt, collection, openai_client, n=5)
                st.session_state.last_retrieved_docs = retrieved
                rag_prompt = build_rag_prompt(
                    user_query=user_prompt,
                    relevant_docs=retrieved,
                    language=language,
                    style=style
                )
                hist = trim_messages_last_n_pairs(st.session_state.messages, 5)
                model_messages = [{"role": "system", "content": rag_prompt}] + hist

                if llm_vendor == "Mistral":
                    response_text = st.write_stream(
                        stream_mistral_response(llm_model, model_messages, MISTRAL_KEY)
                    )
                elif llm_vendor == "Cohere":
                    response_text = st.write_stream(
                        stream_cohere_response(llm_model, model_messages, COHERE_KEY)
                    )
                else:
                    response_text = stream_openai_simple(llm_model, model_messages, OPENAI_KEY)

                final_text = response_text if isinstance(response_text, str) else ""
                st.session_state.messages.append({"role": "assistant", "content": final_text})

            if retrieved:
                with st.expander("Documents used as context"):
                    for i, (doc_id, doc_text, meta) in enumerate(retrieved, start=1):
                        fname = meta.get("filename", "unknown")
                        st.markdown(f"**Doc {i}: {fname} (id={doc_id})**")
                        st.write(doc_text[:800] + ("..." if len(doc_text) > 800 else ""))

