import streamlit as st
import requests

# Using Streamlit secrets for API keys
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY")
COHERE_KEY = st.secrets.get("COHERE_API_KEY")

st.set_page_config(page_title="HW 3 — Multi-LLM Chatbot", layout="wide")
st.title("HW 3: Multi-LLM Streaming Chatbot")

# ========== HELPERS ==========

def style_instruction(style: str) -> str:
    if style == "100 words":
        return "Answer in about 100 words."
    if style == "2 connected paragraphs":
        return "Answer in exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Provide exactly five concise bullet points."
    return ""

def fetch_url_content(url):
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=6)
        if r.ok:
            return r.text[:12000]
        else:
            return f"[Could not load {url}: HTTP {r.status_code}]"
    except Exception as e:
        return f"[Error loading {url}: {e}]"

def trim_messages_last_n_pairs(messages, n_pairs=6):
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

def summarize_conversation(messages, language="English"):
    summary = []
    for msg in messages[-6:]:
        role = {"user": "User", "assistant": "Assistant"}.get(msg["role"], msg["role"])
        summary.append(f"{role}: {msg['content']}")
    return "\n".join(summary)

def trim_messages_to_token_limit(messages, max_tokens=2000):
    approx_char_limit = max_tokens * 4
    chars = 0
    trimmed = []
    for m in reversed(messages):
        chars += len(m.get("content", ""))
        trimmed.append(m)
        if chars >= approx_char_limit:
            break
    return list(reversed(trimmed))

def stream_openai_response(model, messages, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7,
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

# ========== SIDEBAR ==========

st.sidebar.header("Chatbot Controls")

url_1 = st.sidebar.text_input("URL #1 (optional)", key="url_1")
url_2 = st.sidebar.text_input("URL #2 (optional)", key="url_2")

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
            "gpt-4o",        # flagship (default)
            "gpt-4-turbo",
            "gpt-3.5-turbo"
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
        "Buffer: 6 questions",
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

# ========== MAIN ==========

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    base_system = (
        f"Write the output in {language}. "
        "Use simple language suitable for a 10-year-old: short sentences, clear examples. "
        f"{style_instruction(style)}"
    )

    url_text_1 = fetch_url_content(url_1) if url_1 else ""
    url_text_2 = fetch_url_content(url_2) if url_2 else ""
    url_part = ""
    if url_text_1:
        url_part += f"\n\nContent from URL#1:\n{url_text_1[:4000]}"
    if url_text_2:
        url_part += f"\n\nContent from URL#2:\n{url_text_2[:4000]}"
    if url_part:
        base_system += "\nUse information from the following sources if relevant:" + url_part

    if memory_type == "Buffer: 6 questions":
        trimmed_history = trim_messages_last_n_pairs(st.session_state.messages, 6)
    elif memory_type == "Conversation summary":
        summary = summarize_conversation(st.session_state.messages, language)
        trimmed_history = [{"role": "system", "content": f"Summary of previous conversation:\n{summary}"}]
    else:  # Buffer: 2000 tokens
        trimmed_history = trim_messages_to_token_limit(st.session_state.messages, 2000)

    model_messages = [{"role": "system", "content": base_system}] + trimmed_history

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