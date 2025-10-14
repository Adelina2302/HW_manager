import streamlit as st
import os
import sys
import importlib
from io import BytesIO

try:
    sys.modules['sqlite3'] = importlib.import_module('pysqlite3')
except ModuleNotFoundError:
    pass

import chromadb
from openai import OpenAI
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY", "")
COHERE_KEY = st.secrets.get("COHERE_API_KEY", "")

if not OPENAI_KEY:
    st.error("OPENAI_API_KEY is not set! Please configure your key in Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="HW 5: Enhanced iSchool Chatbot", layout="wide")
st.title("HW 5: Enhanced iSchool Chatbot")

CHROMA_DB_PATH = "./ChromaDB_for_hw5"

def get_openai_client():
    return OpenAI(api_key=OPENAI_KEY)

@st.cache_resource(show_spinner=False)
def get_chromadb_collection():
    chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection("HW5Collection")
    return collection

def fixed_size_chunking(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_html_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n")

def create_vector_db_if_needed():
    if not os.path.exists(CHROMA_DB_PATH):
        html_dir = os.path.join(os.path.dirname(__file__), "..", "su_orgs")
        html_dir = os.path.abspath(html_dir)
        if not os.path.exists(html_dir):
            st.error(f"Directory '{html_dir}' does not exist.")
            st.stop()
        html_files = [f for f in os.listdir(html_dir) if f.endswith(".html")]
        chunks = []
        for fname in html_files:
            text = extract_html_text(os.path.join(html_dir, fname))
            doc_chunks = fixed_size_chunking(text, chunk_size=500, overlap=100)
            for i, chunk in enumerate(doc_chunks):
                # No filtering: add all chunks
                chunk_id = f"{fname}_chunk{i+1}"
                chunks.append((chunk_id, chunk))
        chroma_client = chromadb.PersistentClient(CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection("HW5Collection")
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
    else:
        pass

create_vector_db_if_needed()

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

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = get_openai_client()
if 'HW5_vectorDB' not in st.session_state:
    st.session_state.HW5_vectorDB = get_chromadb_collection()
collection = st.session_state.HW5_vectorDB
openai_client = st.session_state.openai_client

st.header("1. Upload PDFs (optional)")
uploaded_files = st.file_uploader(
    "Upload course PDF files",
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
    with st.expander("Current documents in vector database:"):
        for doc_id in all_ids:
            st.write(f"- {doc_id}")
else:
    st.warning("No documents in the vector database yet.")

if st.button("🗑️ Clear the vector database"):
    try:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
            st.success("All files removed.")
        else:
            st.info("The collection is already empty.")
    except Exception as e:
        st.error(f"Error while deleting: {e}")

st.markdown("---")

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

def fetch_relevant_docs(query, collection, openai_client, n=5):
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

def parse_doc_reference(user_input, last_docs):
    import re
    num_map = {
        "first": 0, "1": 0, "one": 0,
        "second": 1, "2": 1, "two": 1,
        "third": 2, "3": 2, "three": 2,
        "fourth": 3, "4": 3, "four": 3,
        "fifth": 4, "5": 4, "five": 4,
        "last": len(last_docs)-1 if last_docs else None,
    }
    for word, idx in num_map.items():
        if word in user_input.lower() and idx is not None and idx < len(last_docs):
            return idx
    for i, (doc_id, _) in enumerate(last_docs):
        if doc_id.lower() in user_input.lower():
            return i
    match = re.search(r"\b(\d+)\b", user_input)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(last_docs):
            return idx
    return None

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
            "gpt-5-nano",
            "gpt-4o",
            "gpt-4-turbo"
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
    st.session_state.last_retrieved_docs = []

st.header("2. Enhanced Chatbot (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about iSchool organizations...")

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

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.awaiting_more_info:
        idx = parse_doc_reference(prompt, st.session_state.last_retrieved_docs)
        if idx is not None:
            doc_id, doc_text = st.session_state.last_retrieved_docs[idx]
            rag_prompt = build_rag_prompt(
                user_query=prompt,
                relevant_docs=[(doc_id, doc_text)],
                language=language,
                style=style
            )
            trimmed_history = trim_messages_last_n_pairs(st.session_state.messages, 5)
            model_messages = [{"role": "system", "content": rag_prompt}] + trimmed_history
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if llm_vendor == "OpenAI":
                        response_text = stream_openai_response(llm_model, model_messages, OPENAI_KEY)
                    elif llm_vendor == "Mistral":
                        response_text = st.write_stream(
                            stream_mistral_response(llm_model, model_messages, MISTRAL_KEY)
                        )
                    elif llm_vendor == "Cohere":
                        response_text = st.write_stream(
                            stream_cohere_response(llm_model, model_messages, COHERE_KEY)
                        )
                    else:
                        response_text = ""
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.awaiting_more_info = False
        else:
            st.warning("Please specify which document you are referring to. Available documents:")
            for i, (doc_id, _) in enumerate(st.session_state.last_retrieved_docs):
                st.write(f"{i+1}. {doc_id}")
    else:
        relevant_docs = fetch_relevant_docs(prompt, collection, openai_client, n=5)
        st.session_state.last_retrieved_docs = relevant_docs
        rag_prompt = build_rag_prompt(
            user_query=prompt,
            relevant_docs=relevant_docs,
            language=language,
            style=style
        )
        if relevant_docs:
            with st.expander("Documents used as context", expanded=False):
                for i, (doc_id, doc_text) in enumerate(relevant_docs):
                    st.markdown(f"**Document {i+1}: {doc_id}**")
                    st.write(doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""))
        trimmed_history = trim_messages_last_n_pairs(st.session_state.messages, 5)
        model_messages = [{"role": "system", "content": rag_prompt}] + trimmed_history
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if llm_vendor == "OpenAI":
                    response_text = stream_openai_response(llm_model, model_messages, OPENAI_KEY)
                elif llm_vendor == "Mistral":
                    response_text = st.write_stream(
                        stream_mistral_response(llm_model, model_messages, MISTRAL_KEY)
                    )
                elif llm_vendor == "Cohere":
                    response_text = st.write_stream(
                        stream_cohere_response(llm_model, model_messages, COHERE_KEY)
                    )
                else:
                    response_text = ""
        st.session_state.messages.append({"role": "assistant", "content": response_text})

        if len(relevant_docs) > 1:
            st.session_state.awaiting_more_info = True