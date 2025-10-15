
import sqlite3  # noqa: F401
import os
import json
import time
import re
import hashlib
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Third-party
import streamlit as st
import pandas as pd
import chromadb
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- Configuration ----------------
CSV_FILENAME = "Example_news_info_for_testing.csv"
CHROMA_PATH = "./news_chroma_db"
COLLECTION_NAME = "NewsCollection"
EMBEDDING_MODEL = "text-embedding-3-small"
PAGE_CACHE_DIR = "./page_cache"

# Chunking defaults
CHUNK_SIZE_DEFAULT = 300
CHUNK_OVERLAP_DEFAULT = 60
MIN_ARTICLE_LEN_DEFAULT = 150
FETCH_TIMEOUT_DEFAULT = 20
SLEEP_BETWEEN_REQUESTS = 0.10
USER_AGENT = "Mozilla/5.0 (compatible; HW7NewsBot/1.0; +https://example.com/bot)"

AUTO_INDEX_ON_START = True  # auto-ingest if DB is empty

# Conversation memory defaults
MEMORY_MAX_USER_TURNS_DEFAULT = 5      # Buffer: last 5 questions
MEMORY_TOKEN_BUDGET_DEFAULT = 2000     # ~2000 tokens
SUMMARY_MAX_WORDS = 160                # target words for running summary

# Strict law-only is used ONLY for "most interesting" ranking
LEGAL_ONLY_FOR_INTERESTING = True

# Legal domain signals
LEGAL_KEYWORDS = [
    "regulation","regulatory","antitrust","competition","litigation","lawsuit",
    "settlement","class action","injunction","compliance","aml","anti-money laundering",
    "kyc","enforcement","sanctions","gdpr","privacy","data protection","merger","m&a",
    "acquisition","doj","ftc","sec","fca","ec","ecb","esma","ofac","bribery","fines",
    "penalty","consent order","whistleblower","investigation","probe"
]
LEGAL_CATEGORY_WEIGHTS = {
    "regulation": 1.0,
    "antitrust": 0.95,
    "litigation": 0.9,
    "compliance": 0.85,
    "m&a": 0.82,
    "governance": 0.75,
    "privacy": 0.75,
    "finance": 0.6,
    "technology": 0.55,
    "other": 0.4
}

# Models behavior flags
OPENAI_TOOL_MODELS = {"gpt-4o", "gpt-4o-mini"}  # supports Chat Completions tools
OPENAI_RESPONSES_MODELS = {"gpt-5-nano"}       # prefer Responses API (no tools)

st.set_page_config(page_title="HW7 Auto News RAG (Redesigned)", layout="wide")
st.title("📰 HW7 Auto News RAG — OpenAI Tools / Mistral No-Tools (Redesigned)")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY", os.environ.get("MISTRAL_API_KEY", ""))

if not OPENAI_KEY:
    st.warning("OPENAI_API_KEY not found — embeddings and OpenAI chat/Responses may not work.")
if not MISTRAL_KEY:
    st.info("MISTRAL_API_KEY not set — Mistral path is optional.")

os.makedirs(PAGE_CACHE_DIR, exist_ok=True)

# ---------------- Chroma Init ----------------
@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

collection = get_chroma_collection()

# ---------------- OpenAI Client / Embeddings ----------------
def get_openai_client():
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY is missing.")
    if OpenAI is None:
        raise ImportError("openai package is not installed.")
    return OpenAI(api_key=OPENAI_KEY)

def get_embedding(text: str):
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

# ---------------- OpenAI safe wrapper (chat.completions or Responses fallback) ----------------
def _combine_messages_for_prompt(system_prompt: str, memory_messages: List[Dict[str, str]], user_text: str) -> str:
    lines = []
    if system_prompt:
        lines.append(f"[System]\n{system_prompt}")
    if memory_messages:
        lines.append("[Memory]")
        for m in memory_messages:
            role = m.get("role","")
            content = m.get("content","")
            if role == "system":
                lines.append(f"(Summary) {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
    lines.append(f"[User]\n{user_text}")
    return "\n\n".join(lines)

def openai_chat_or_responses_generate(
    client,
    *,
    model: str,
    system_prompt: str,
    memory_messages: List[Dict[str, str]],
    user_text: str,
    tools=None,
    tool_choice=None,
    temperature: Optional[float] = None,
    max_new_tokens: int = 900,
) -> Tuple[str, Optional[Any]]:
    """
    Tries Chat Completions first (with tools if provided). If model/params unsupported,
    falls back to Responses API by collapsing content into a single prompt text.

    Returns (text, raw_response_or_None).
    """
    # Prefer Chat Completions if the model is known to support it (or tools provided)
    try_chat = (model in OPENAI_TOOL_MODELS) or (tools is None and model not in OPENAI_RESPONSES_MODELS)

    if try_chat:
        # Build Chat messages
        chat_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        chat_messages += memory_messages
        chat_messages += [{"role": "user", "content": user_text}]

        def _call_chat(use_max_completion_tokens: bool, include_temperature: bool):
            kwargs = {
                "model": model,
                "messages": chat_messages,
            }
            if tools is not None:
                kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
            if include_temperature and (temperature is not None):
                kwargs["temperature"] = temperature
            # token param
            if use_max_completion_tokens:
                kwargs["max_completion_tokens"] = max_new_tokens
            else:
                kwargs["max_tokens"] = max_new_tokens
            return client.chat.completions.create(**kwargs)

        # Try chat path with progressive fallbacks for tokens/temperature keys
        try:
            resp = _call_chat(use_max_completion_tokens=False, include_temperature=True)
            content = resp.choices[0].message.content or ""
            # If we used tools, the content may be empty; caller should handle tool loop separately
            return content, resp
        except Exception as e1:
            msg1 = str(e1)
            # Switch to max_completion_tokens
            if ("max_tokens" in msg1 and ("Unsupported" in msg1 or "Unrecognized" in msg1)):
                try:
                    resp = _call_chat(use_max_completion_tokens=True, include_temperature=True)
                    return (resp.choices[0].message.content or ""), resp
                except Exception as e1b:
                    msg1b = str(e1b)
                    if "temperature" in msg1b and ("Unsupported" in msg1b or "Unsupported parameter" in msg1b):
                        resp = _call_chat(use_max_completion_tokens=True, include_temperature=False)
                        return (resp.choices[0].message.content or ""), resp
                    # otherwise fall through to Responses
            # Temperature unsupported -> drop it and retry once with max_tokens
            if ("temperature" in msg1 and ("Unsupported" in msg1 or "Unsupported parameter" in msg1)):
                try:
                    resp = _call_chat(use_max_completion_tokens=False, include_temperature=False)
                    return (resp.choices[0].message.content or ""), resp
                except Exception:
                    pass
            # If error suggests Responses API, we fall through

    # Fallback to Responses API
    try:
        prompt = _combine_messages_for_prompt(system_prompt, memory_messages, user_text)
        # responses.create has a different schema vs chat
        # IMPORTANT: set lower temperature default, request low reasoning effort to save tokens
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=(0.2 if temperature is None else float(temperature)),
            max_output_tokens=max_new_tokens,
            reasoning={"effort": "low"}
        )
        # Robust extraction across SDK variants
        text = ""
        # Newer SDKs expose .output_text
        if hasattr(resp, "output_text") and resp.output_text:
            text = resp.output_text
        else:
            # Older structure: resp.output[0].content[0].text
            try:
                if hasattr(resp, "output") and resp.output:
                    # output is a list of messages/segments
                    chunks = []
                    for o in resp.output:
                        # o may have 'content' list with dicts carrying 'text'
                        if hasattr(o, "content") and o.content:
                            for c in o.content:
                                if hasattr(c, "text") and c.text:
                                    chunks.append(c.text)
                                elif isinstance(c, dict) and "text" in c:
                                    chunks.append(str(c["text"]))
                        # Some SDKs: o has 'text' directly
                        if hasattr(o, "text") and o.text:
                            chunks.append(o.text)
                    text = "\n".join(chunks).strip()
            except Exception:
                pass

        # Detect incomplete due to max_output_tokens and do an automatic concise retry
        try:
            is_incomplete = getattr(resp, "status", "") == "incomplete"
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
        except Exception:
            is_incomplete, reason = False, None

        if (not text) and is_incomplete and (reason == "max_output_tokens"):
            short_prompt = prompt + "\n\nInstruction: Continue concisely in <=120 words."
            resp2 = client.responses.create(
                model=model,
                input=short_prompt,
                temperature=0.1,
                max_output_tokens=min(2000, int(max_new_tokens * 1.5)),
                reasoning={"effort": "low"}
            )
            if hasattr(resp2, "output_text") and resp2.output_text:
                return resp2.output_text, resp2
            # Repeat legacy extraction
            try:
                chunks2 = []
                if hasattr(resp2, "output") and resp2.output:
                    for o in resp2.output:
                        if hasattr(o, "content") and o.content:
                            for c in o.content:
                                if hasattr(c, "text") and c.text:
                                    chunks2.append(c.text)
                                elif isinstance(c, dict) and "text" in c:
                                    chunks2.append(str(c["text"]))
                        if hasattr(o, "text") and o.text:
                            chunks2.append(o.text)
                text2 = "\n".join(chunks2).strip()
                if text2:
                    return text2, resp2
            except Exception:
                pass

        if not text:
            # Last-resort stringify
            text = str(resp)
        return text, resp
    except Exception as e:
        return f"[OpenAI Responses API error: {e}]", None

# ---------------- Helpers: Signals, Hash, Cache, Extract ----------------
def legal_signals(meta: Dict[str, Any], doc: str) -> Tuple[int, float]:
    txt_low = (doc or "").lower()
    kw_hits = sum(1 for kw in LEGAL_KEYWORDS if kw in txt_low)
    cat = str(meta.get("category","")).lower() if meta else ""
    cat_weight = LEGAL_CATEGORY_WEIGHTS.get(cat, 0.0)
    return kw_hits, cat_weight

def is_legal(meta: Dict[str, Any], doc: str) -> bool:
    kw_hits, cat_weight = legal_signals(meta, doc)
    return (kw_hits > 0) or (cat_weight >= 0.75)

def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]

def cache_path_for(url: str) -> str:
    return os.path.join(PAGE_CACHE_DIR, f"{hash_url(url)}.json")

def fetch_page(url: str, timeout: int) -> Optional[str]:
    cpath = cache_path_for(url)
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("html")
        except Exception:
            pass
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        html = r.text
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump({"html": html, "fetched_at": time.time()}, f)
        return html
    except Exception:
        return None

def clean_text_blocks(blocks: List[str], min_len: int = 40) -> List[str]:
    cleaned = []
    for b in blocks:
        b2 = re.sub(r'\s+', ' ', b).strip()
        if len(b2) >= min_len:
            cleaned.append(b2)
    return cleaned

def extract_main_text_basic(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    for t in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        t.decompose()
    texts = []
    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        txt = tag.get_text(separator=" ", strip=True)
        if txt:
            texts.append(txt)
    texts = clean_text_blocks(texts)
    body = "\n".join(texts)
    return {"title": page_title, "body_plain": body}

# ---------------- Chunking & Indexing ----------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    overlap = min(overlap, max(0, chunk_size - 1))
    res = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        res.append(text[start:end])
        if end >= L:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return res

def index_articles(
    articles: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_article_len: int
) -> int:
    try:
        existing_ids = set(collection.get().get("ids", []))
    except Exception:
        existing_ids = set()
    added = 0
    for art in articles:
        full_text = art["full_text"]
        if len(full_text.strip()) == 0:
            continue
        if len(full_text) < min_article_len:
            chunks = [full_text]
        else:
            chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

        for ci, ch in enumerate(chunks):
            if not ch.strip():
                continue
            doc_id = f"{art['base_id']}__chunk_{ci}"
            if doc_id in existing_ids:
                continue
            try:
                emb = get_embedding(ch)
            except Exception as e:
                st.error(f"Embedding error (url={art.get('url','')} chunk={ci}): {e}")
                continue
            meta = {
                "title": art.get("title",""),
                "date": art.get("date",""),
                "company_name": art.get("company_name",""),
                "url": art.get("url",""),
                "category": art.get("category",""),
                "chunk_index": ci,
                "total_chunks": len(chunks),
                "source_type": art.get("source_type","direct")
            }
            collection.add(
                ids=[doc_id],
                documents=[ch],
                embeddings=[emb],
                metadatas=[meta]
            )
            added += 1
    return added

# ---------------- Retrieval (General and Legal-only variants) ----------------
def retrieve_top_k(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not OPENAI_KEY:
        return []
    try:
        q_emb = get_embedding(query)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = res.get("documents", [[]])[0]
    mets = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    out = []
    for i, doc in enumerate(docs):
        meta = mets[i] if i < len(mets) else {}
        out.append({
            "id": ids[i],
            "title": meta.get("title",""),
            "date": meta.get("date",""),
            "company_name": meta.get("company_name",""),
            "category": meta.get("category",""),
            "url": meta.get("url",""),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "text": doc
        })
    return out

def retrieve_top_k_legal(query: str, k: int = 5, oversample: int = 40) -> List[Dict[str, Any]]:
    if not OPENAI_KEY:
        return []
    try:
        q_emb = get_embedding(query)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []
    res = collection.query(query_embeddings=[q_emb], n_results=max(k, oversample))
    docs = res.get("documents", [[]])[0]
    mets = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    legal = []
    for i in range(len(ids)):
        doc = docs[i]
        meta = mets[i] if i < len(mets) else {}
        if is_legal(meta, doc):
            legal.append({
                "id": ids[i],
                "title": meta.get("title",""),
                "date": meta.get("date",""),
                "company_name": meta.get("company_name",""),
                "category": meta.get("category",""),
                "url": meta.get("url",""),
                "chunk_index": meta.get("chunk_index"),
                "total_chunks": meta.get("total_chunks"),
                "text": doc
            })
        if len(legal) >= k:
            break
    return legal

def retrieve_unique_news_by_url_general(query: str, k: int = 5, oversample: int = 40) -> List[Dict[str, Any]]:
    """
    General (not law-only): get top results, then de-duplicate by URL so each item is a distinct news article.
    """
    if not OPENAI_KEY:
        return []
    try:
        q_emb = get_embedding(query)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []
    res = collection.query(query_embeddings=[q_emb], n_results=max(k, oversample))
    docs = res.get("documents", [[]])[0]
    mets = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    seen = set()
    items = []
    for i in range(len(ids)):
        meta = mets[i] if i < len(mets) else {}
        url = meta.get("url","")
        if not url or url in seen:
            continue
        seen.add(url)
        doc = docs[i]
        items.append({
            "title": meta.get("title","") or "Untitled",
            "date": meta.get("date",""),
            "company_name": meta.get("company_name",""),
            "category": meta.get("category",""),
            "url": url,
            "snippet": doc[:400] + ("..." if len(doc) > 400 else ""),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks")
        })
        if len(items) >= k:
            break
    return items

# ---------------- Conversation Memory ----------------
def approx_token_count(text: str) -> int:
    # Rough heuristic: ~4 chars per token
    return max(1, len(text) // 4)

def build_memory_messages() -> List[Dict[str, str]]:
    """
    Build memory messages to prepend to the LLM call, based on UI-selected mode.
    Modes (mutually exclusive):
      - "Buffer: 5 questions" (actually user-chosen N)
      - "Conversation summary"
      - "Buffer: 2000 tokens" (user-chosen token budget)
    """
    mem_mode = st.session_state.get("mem_mode", "Buffer: 5 questions")
    mem_msgs: List[Dict[str, str]] = []

    history = st.session_state.get("chat_history", [])

    if mem_mode == "Conversation summary":
        summary = st.session_state.get("conversation_summary", "").strip()
        if summary:
            mem_msgs.append({"role": "system", "content": f"Conversation summary so far:\n{summary}"})
        return mem_msgs

    # Build (user, assistant) pairs
    pairs = []
    current_user = None
    for msg in history:
        if msg["role"] == "user":
            current_user = msg["content"]
        elif msg["role"] == "assistant" and current_user is not None:
            pairs.append((current_user, msg["content"]))
            current_user = None

    if mem_mode == "Buffer: 5 questions":
        last_n = st.session_state.get("mem_last_n", MEMORY_MAX_USER_TURNS_DEFAULT)
        pairs = pairs[-last_n:]
        for u, a in pairs:
            mem_msgs.append({"role": "user", "content": u})
            mem_msgs.append({"role": "assistant", "content": a})
        return mem_msgs

    if mem_mode == "Buffer: 2000 tokens":
        token_budget = st.session_state.get("mem_token_budget", MEMORY_TOKEN_BUDGET_DEFAULT)
        # consume from the end (most recent)
        pairs = pairs[::-1]
        used = 0
        selected = []
        for u, a in pairs:
            cost = approx_token_count(u) + approx_token_count(a)
            if used + cost > token_budget:
                break
            selected.append((u, a))
            used += cost
        # reverse back to chronological order
        for u, a in reversed(selected):
            mem_msgs.append({"role": "user", "content": u})
            mem_msgs.append({"role": "assistant", "content": a})
        return mem_msgs

    return mem_msgs

def update_conversation_summary(latest_user: str, latest_assistant: str):
    """
    Update running summary only when the selected mode is "Conversation summary".
    Uses the currently selected vendor+model from the sidebar:
      - If vendor == Mistral: use selected mistral model (large/medium)
      - If vendor == OpenAI: use selected OpenAI model (tools off, single-shot)
    """
    if st.session_state.get("mem_mode") != "Conversation summary":
        return

    prev_summary = st.session_state.get("conversation_summary", "")
    prompt = (
        "You are a helpful assistant that maintains a concise running summary of a conversation. "
        f"Rewrite/extend the summary to at most {SUMMARY_MAX_WORDS} words, keeping key intents, decisions, and follow-ups.\n\n"
        f"Previous summary:\n{prev_summary}\n\n"
        f"Latest user message:\n{latest_user}\n\n"
        f"Latest assistant message:\n{latest_assistant}\n\n"
        "Return only the updated summary without extra commentary."
    )

    vendor_sel = st.session_state.get("vendor", "OpenAI")

    # Build memory (no need, we already include prev summary + latests)
    try:
        if vendor_sel == "Mistral" and MISTRAL_KEY:
            mistral_model = st.session_state.get("mistral_model_choice", "mistral-large-latest")
            r = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_KEY}"},
                json={
                    "model": mistral_model,
                    "messages": [
                        {"role": "system", "content": "Summarize and update concisely."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 256
                },
                timeout=30
            )
            if r.ok:
                data = r.json()
                st.session_state.conversation_summary = (data["choices"][0]["message"]["content"] or "").strip()
                return

        if OPENAI_KEY and OpenAI is not None:
            client = get_openai_client()
            openai_model = st.session_state.get("openai_model_choice", "gpt-4o")
            # Use Responses fallback for models like gpt-5-nano
            txt, _ = openai_chat_or_responses_generate(
                client,
                model=openai_model,
                system_prompt="Summarize and update concisely.",
                memory_messages=[],
                user_text=prompt,
                tools=None,
                tool_choice=None,
                temperature=0.2,
                max_new_tokens=256
            )
            st.session_state.conversation_summary = (txt or "").strip()
            return
    except Exception:
        pass

    # Fallback concise truncation
    combined = (prev_summary + "\n" + latest_user + "\n" + latest_assistant).strip()
    st.session_state.conversation_summary = combined[:1200]

# ---------------- Session state init ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "last_ranked_list" not in st.session_state:
    st.session_state.last_ranked_list = []
if "last_relevant_list" not in st.session_state:
    st.session_state.last_relevant_list = []
if "last_answer_context" not in st.session_state:
    st.session_state.last_answer_context = {}
if "vendor" not in st.session_state:
    st.session_state.vendor = "OpenAI"

# ---------------- Tools (Function Calling) ----------------
def style_instruction(style: str) -> str:
    return {
        "Normal": "",
        "100 words": "Answer in about 100 words.",
        "2 connected paragraphs": "Answer in exactly two connected paragraphs.",
        "5 bullet points": "Provide exactly five concise bullet points."
    }.get(style, "")

def collect_chunks_for_url(target_url: str, max_chunks: int = 6) -> List[Dict[str, Any]]:
    try:
        data = collection.get(include=["metadatas", "documents", "ids"])
    except Exception:
        return []
    mets = data.get("metadatas", [])
    docs = data.get("documents", [])
    ids = data.get("ids", [])
    rows = []
    for i in range(len(ids)):
        m = mets[i]
        if str(m.get("url","")).strip() == str(target_url).strip():
            rows.append({
                "id": ids[i],
                "title": m.get("title",""),
                "date": m.get("date",""),
                "company_name": m.get("company_name",""),
                "category": m.get("category",""),
                "url": m.get("url",""),
                "chunk_index": m.get("chunk_index", 0),
                "total_chunks": m.get("total_chunks", None),
                "text": docs[i]
            })
    rows_sorted = sorted(rows, key=lambda r: (r.get("chunk_index") if r.get("chunk_index") is not None else 0))
    return rows_sorted[: max_chunks]

def legal_rank_interesting_news_list(limit: int = 5) -> List[Dict[str, Any]]:
    data = collection.get(include=["metadatas","documents"])
    mets = data.get("metadatas", [])
    docs = data.get("documents", [])
    now = datetime.utcnow()
    scored = []
    for meta, doc in zip(mets, docs):
        if LEGAL_ONLY_FOR_INTERESTING and not is_legal(meta, doc):
            continue
        recency = 0.5
        date_raw = meta.get("date","")
        if date_raw:
            try:
                d = datetime.fromisoformat(date_raw.replace("Z","").split("+")[0])
                age_days = (now - d).days
                recency = max(0.0, 1 - age_days/365)
            except Exception:
                pass
        length_score = min(len(doc)/1500, 1.0)
        first_chunk_bonus = 1.0 if meta.get("chunk_index",0) == 0 else 0.6
        kw_hits, cat_weight = legal_signals(meta, doc)
        legal_keyword_boost = min(0.12 * kw_hits, 0.72)
        category_weight = cat_weight if meta.get("category") else (0.5 if kw_hits > 0 else 0.3)
        score = 0.40*recency + 0.25*legal_keyword_boost + 0.20*length_score + 0.10*first_chunk_bonus + 0.05*category_weight
        scored.append((score, meta, doc))
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    seen_urls = set()
    items = []
    for sc, m, dtext in ranked:
        url = m.get("url","")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append({
            "title": m.get("title",""),
            "date": m.get("date",""),
            "company_name": m.get("company_name",""),
            "category": m.get("category",""),
            "url": url,
            "score": round(sc,4),
            "snippet": dtext[:250] + ("..." if len(dtext) > 250 else "")
        })
        if len(items) >= max(1, min(10, limit)):
            break
    return items

def tool_find_relevant_news(topic: str, limit: int = 5) -> str:
    found = retrieve_top_k(topic, 12)
    seen = set()
    ui_list = []
    for f in found:
        url = f.get("url","")
        if not url or url in seen:
            continue
        seen.add(url)
        ui_list.append({
            "title": f.get("title",""),
            "date": f.get("date",""),
            "company_name": f.get("company_name",""),
            "category": f.get("category",""),
            "url": url,
            "snippet": f.get("text","")[:300] + ("..." if len(f.get("text","")) > 300 else "")
        })
        if len(ui_list) >= limit:
            break
    st.session_state.last_relevant_list = ui_list[:]
    return json.dumps(ui_list, ensure_ascii=False, indent=2)

def tool_get_news_details(index: int, source_list: str = "auto", max_chunks: int = 6) -> str:
    ranked = st.session_state.get("last_ranked_list", [])
    relevant = st.session_state.get("last_relevant_list", [])
    chosen_name, chosen_list = None, []
    if source_list == "ranked":
        chosen_name, chosen_list = "ranked", ranked
    elif source_list == "relevant":
        chosen_name, chosen_list = "relevant", relevant
    else:
        if ranked:
            chosen_name, chosen_list = "ranked", ranked
        elif relevant:
            chosen_name, chosen_list = "relevant", relevant
        else:
            return json.dumps({"status":"need_clarification","reason":"no_recent_list","message":"Ask me to list news first."}, ensure_ascii=False)
    if index < 1 or index > len(chosen_list):
        return json.dumps({"status":"need_clarification","reason":"index_out_of_range","message":f"Pick 1..{len(chosen_list)} from last {chosen_name} list."}, ensure_ascii=False)
    item = chosen_list[index-1]
    url = item.get("url","")
    if not url:
        return json.dumps({"status":"need_clarification","reason":"missing_url","message":"No URL for that item."}, ensure_ascii=False)
    chunks = collect_chunks_for_url(url, max_chunks=max_chunks)
    return json.dumps({
        "status":"ok",
        "source_list": chosen_name,
        "index": index,
        "item": {
            "title": item.get("title",""),
            "date": item.get("date",""),
            "company_name": item.get("company_name",""),
            "category": item.get("category",""),
            "url": url
        },
        "chunks": [
            {"chunk_index": c.get("chunk_index"), "total_chunks": c.get("total_chunks"), "text": c.get("text","")[:1200]}
            for c in chunks
        ]
    }, ensure_ascii=False, indent=2)

def tool_prepare_answer_context(query: str, top_k: int = 5, max_chunks_per_source: int = 2) -> str:
    """
    GENERAL (not law-only): build source-backed context for a formatted answer.
    """
    items = retrieve_top_k(query, k=top_k*3)
    by_url: Dict[str, Dict[str, Any]] = {}
    for it in items:
        url = it.get("url","")
        if not url:
            continue
        if url not in by_url:
            by_url[url] = {
                "title": it.get("title",""),
                "date": it.get("date",""),
                "company_name": it.get("company_name",""),
                "category": it.get("category",""),
                "url": url,
                "snippets": [],
            }
        by_url[url]["snippets"].append({
            "chunk_index": it.get("chunk_index"),
            "total_chunks": it.get("total_chunks"),
            "text": it.get("text","")
        })
    sources = []
    for url, meta in by_url.items():
        snips = sorted(meta["snippets"], key=lambda s: (s.get("chunk_index") if s.get("chunk_index") is not None else 0))
        trimmed = []
        for s in snips[:max_chunks_per_source]:
            trimmed.append({
                "chunk_index": s.get("chunk_index"),
                "total_chunks": s.get("total_chunks"),
                "text": s.get("text","")[:1200]
            })
        sources.append({
            "title": meta["title"],
            "date": meta["date"],
            "company_name": meta["company_name"],
            "category": meta["category"],
            "url": url,
            "snippets": trimmed
        })
    sources = sources[:top_k]
    payload = {
        "status": "ok" if sources else "no_results",
        "question": query,
        "sources": [dict(item, id=i+1) for i, item in enumerate(sources)]
    }
    st.session_state.last_answer_context = payload
    return json.dumps(payload, ensure_ascii=False, indent=2)

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "find_relevant_news",
            "description": "Find general news items about a specific topic using semantic retrieval (deduplicated by URL).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic or keyword"},
                    "limit": {"type": "integer", "description": "How many items (1-10).", "default": 5}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_interesting_news",
            "description": "Return a ranked list of law-related news items (legal-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many items (1-10).", "default": 5}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_details",
            "description": "Return detailed context for a specific item by its 1-based index from the last list (ranked or relevant).",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer","minimum": 1},
                    "source_list": {"type": "string","enum": ["auto","ranked","relevant"],"default":"auto"},
                    "max_chunks": {"type":"integer","minimum":1,"maximum":20,"default":6}
                },
                "required": ["index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_answer_context",
            "description": "Build a general (not law-only) source context for formatted answers with citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer","default":5,"minimum":1,"maximum":10},
                    "max_chunks_per_source": {"type": "integer","default":2,"minimum":1,"maximum":6}
                },
                "required": ["query"]
            }
        }
    }
]

def dispatch_tool_call(name: str, args: Dict[str, Any]) -> str:
    if name == "find_relevant_news":
        topic = args.get("topic", "")
        limit = int(args.get("limit", 5))
        return tool_find_relevant_news(topic, limit)
    if name == "rank_interesting_news":
        limit = int(args.get("limit", 5))
        items = legal_rank_interesting_news_list(limit)
        st.session_state.last_ranked_list = items[:]
        return json.dumps(items, ensure_ascii=False, indent=2)
    if name == "get_news_details":
        idx = int(args.get("index", 1))
        source_list = args.get("source_list", "auto")
        max_chunks = int(args.get("max_chunks", 6))
        return tool_get_news_details(idx, source_list, max_chunks)
    if name == "prepare_answer_context":
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))
        mcs = int(args.get("max_chunks_per_source", 2))
        return tool_prepare_answer_context(query, top_k, mcs)
    return "Tool not recognized"

# ---------------- Intent helpers ----------------
def parse_index_from_text(text: str) -> Optional[int]:
    m = re.search(r"#\s*(\d+)", text or "")
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"\b(\d{1,2})\b", text or "")
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            pass
    return None

def is_most_interesting_query(q: str) -> bool:
    ql = (q or "").lower()
    triggers = [
        # EN
        "most interesting",
        "most-interesting",
        "most interesting news",
        "most interesting information",
        # RU
        "самая интересная новость",
        "самые интересные новости",
        "наиболее интересные новости",
        "самое интересное",
        "самая интересная информация",
        "топ интересных новостей",
        "топ-5 интересных новостей",
        # ES/DE examples
        "noticias más interesantes",
        "interessantesten nachrichten",
    ]
    return any(t in ql for t in triggers)

def infer_intent(user_query: str) -> str:
    q = (user_query or "").lower()
    if is_most_interesting_query(q):
        return "interesting"
    idx = parse_index_from_text(user_query)
    if idx:
        return "details"
    if "find news about" in q or "news about" in q or "find news" in q:
        return "topic"
    if "найди новости" in q or "новости про" in q or "новости о" in q:
        return "topic"
    return "qa"

def build_context_for_non_tool(user_query: str) -> Tuple[str, str]:
    """
    Returns (intent, context_text) for models without tools (Mistral; OpenAI gpt-5-nano).
    """
    intent = infer_intent(user_query)
    context_blocks: List[str] = []

    if intent == "interesting":
        items = legal_rank_interesting_news_list(limit=5)
        st.session_state.last_ranked_list = items[:]
        for i, it in enumerate(items, 1):
            context_blocks.append(f"[{i}] {it.get('title','')} | {it.get('date','')} | {it.get('url','')}\n{it.get('snippet','')}")
        return intent, "\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT"

    if intent == "details":
        idx = parse_index_from_text(user_query)
        lst = st.session_state.last_ranked_list or st.session_state.last_relevant_list
        if lst and idx and 1 <= idx <= len(lst):
            url = lst[idx-1].get("url","")
            chunks = collect_chunks_for_url(url, max_chunks=6)
            for c in chunks:
                context_blocks.append(
                    f"[Chunk {c.get('chunk_index')}/{c.get('total_chunks')}] {c.get('text','')[:800]}"
                )
        return intent, "\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT"

    if intent == "topic":
        m = re.search(r"(?:find\s+news\s+about|news\s+about|find\s+news)\s*[:\-]?\s*(.*)", user_query, flags=re.I)
        topic = (m.group(1).strip() if m and m.group(1) else user_query).strip()
        items = retrieve_unique_news_by_url_general(topic, k=5, oversample=40)
        st.session_state.last_relevant_list = [
            {"title": it["title"], "date": it["date"], "company_name": it["company_name"], "category": it["category"], "url": it["url"], "snippet": it["snippet"]}
            for it in items
        ]
        for i, it in enumerate(items, 1):
            context_blocks.append(f"[{i}] {it.get('title','')} | {it.get('date','')} | {it.get('url','')}\n{it.get('snippet','')}")
        return intent, "\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT"

    # qa fallback: general retrieval
    retrieved = retrieve_top_k(user_query, 5)
    for r in retrieved:
        context_blocks.append(
            f"[{r.get('title','')} | {r.get('date','')} | {r.get('url','')} | Chunk {r.get('chunk_index')}/{r.get('total_chunks')}] {r.get('text','')[:600]}"
        )
    return "qa", "\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT"

def model_supports_tools(model: str) -> bool:
    # ONLY OpenAI tool-capable chat models
    return model in OPENAI_TOOL_MODELS

# ---------------- OpenAI Function Calling (tools) + Responses fallback ----------------
def openai_function_call(system_prompt: str, user_prompt: str, model: str) -> str:
    if not OPENAI_KEY:
        return "OPENAI_API_KEY is missing."
    client = get_openai_client()

    memory_messages = build_memory_messages()
    base_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *memory_messages,
        {"role": "user", "content": user_prompt},
    ]

    # Tools ONLY for supported OpenAI models
    if model_supports_tools(model):
        messages: List[Dict[str, Any]] = list(base_messages)
        for _ in range(3):
            try:
                # First call: let model decide to call tools or answer
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS_OPENAI,
                    tool_choice="auto",
                )
            except Exception:
                break

            msg = resp.choices[0].message
            if getattr(msg, "content", None):
                return msg.content

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        } for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    try:
                        tool_args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        tool_args = {}
                    tool_output = dispatch_tool_call(tc.function.name, tool_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": tool_output,
                    })
                continue
            break  # No tool_calls and no content => break

        # One plain fallback without tools (Chat Completions)
        try:
            plain = client.chat.completions.create(
                model=model,
                messages=base_messages,
            )
            content = plain.choices[0].message.content
            if content:
                return content
        except Exception:
            pass

        # Final fallback to Responses API
        text, _ = openai_chat_or_responses_generate(
            client,
            model=model,
            system_prompt=system_prompt,
            memory_messages=memory_messages,
            user_text=user_prompt,
            tools=None,
            tool_choice=None,
            temperature=0.2,
            max_new_tokens=900
        )
        return text or "I could not generate a response."

   
    intent, context_text = build_context_for_non_tool(user_prompt)
    sys = (
        system_prompt
        + "\nWhen context is provided, ground your answer in that context. "
        + "If the context is insufficient, ask the user to refine."
        + "\nAnswer concisely. Keep the final answer under ~150 words unless explicitly asked otherwise."
    )
    user_text = f"Context:\n{context_text}\n\nUser question: {user_prompt}"

    text, _ = openai_chat_or_responses_generate(
        client,
        model=model,
        system_prompt=sys,
        memory_messages=memory_messages,
        user_text=user_text,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_new_tokens=1200
    )
    return text or "I could not generate a response. Please try rephrasing or check API/model settings."
# ---------------- Mistral RAG (NO tools; app orchestrates, model formats) ----------------
def mistral_rag_answer(user_query: str, language: str, style_txt: str, model: str) -> str:
    """
    Mistral NEVER uses function calling here.
    The app:
      - detects intent,
      - deterministically builds legal-only top-5 list for 'most interesting',
      - gathers chunks for '#N' details if needed,
      - general retrieval otherwise,
    and then asks the model to format the answer nicely.
    """
    if not MISTRAL_KEY:
        return "Mistral key is missing."
    if not OPENAI_KEY:
        return "OpenAI key is required for semantic retrieval."

    # Conversation memory as plain text (optional context)
    memory_messages = build_memory_messages()
    pairs_text = ""
    for m in memory_messages:
        if m["role"] == "system":
            pairs_text += f"[Summary]\n{m['content']}\n\n"
        elif m["role"] in ("user", "assistant"):
            role = "User" if m["role"] == "user" else "Assistant"
            pairs_text += f"{role}: {m['content']}\n"

    intent, base_context_text = build_context_for_non_tool(user_query)

    # For "most interesting": enrich context with chunks for each selected URL (for better grounding)
    context_blocks = []
    if intent == "interesting" and base_context_text != "NO_CONTEXT":
        items = st.session_state.get("last_ranked_list", [])
        for i, it in enumerate(items, 1):
            url = it.get("url", "")
            title = it.get("title", "")
            date = it.get("date", "")
            snippet = it.get("snippet", "")
            chunks = collect_chunks_for_url(url, max_chunks=4) if url else []
            if chunks:
                for c in chunks:
                    context_blocks.append(
                        f"[{i}] Title: {title} | Date: {date} | URL: {url} | "
                        f"Chunk {c.get('chunk_index')}/{c.get('total_chunks')}]\n{c.get('text','')[:750]}"
                    )
            else:
                context_blocks.append(
                    f"[{i}] Title: {title} | Date: {date} | URL: {url}]\n{snippet[:750]}"
                )
        context_text = "\n\n".join(context_blocks) if context_blocks else base_context_text
    else:
        context_text = base_context_text

    # Build instruction
    base_instruction = (
        f"Respond in {language}. {style_txt}\n"
        "When context is provided, ground your answer strictly in that context; if insufficient, ask the user to refine.\n"
        "Use clear headings and bullets, and include succinct source mentions when available."
    )

    if intent == "interesting":
        format_instruction = (
            "Task: From the provided context, produce a concise, well-structured brief of the MOST INTERESTING legal-related news (top 5).\n"
            "Requirements:\n"
            "- Start with a short headline for the set.\n"
            "- Then output a numbered list 1..5. For each item include: Title, 2–3 sentence summary, Date, and a 'Read more' source label.\n"
            "- Close with 1–2 sentences synthesizing the overall trend.\n"
            "Ground strictly in the given context. If some items lack enough info, say so briefly."
        )
    elif intent == "details":
        format_instruction = (
            "Produce a focused analysis with:\n"
            "1) Title\n2) What happened\n3) Why it matters\n4) Key excerpts (quoted)\n"
            "5) Implications and next steps\nReference sources inline when appropriate."
        )
    else:
        format_instruction = (
            "Structure your answer with:\n"
            "1) Title\n2) Executive summary\n3) Key points (bulleted)\n"
            "4) Implications for decision-makers (bulleted)\n"
            "5) Risks & considerations (bulleted)\n"
            "6) Timeline/next steps (if relevant)\nReference sources inline when appropriate."
        )

    prompt = (
        f"{base_instruction}\n{format_instruction}\n\n"
        f"Conversation memory (optional):\n{pairs_text}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User request: {user_query}"
    )
    try:
        mistral_model = st.session_state.get("mistral_model_choice", model)
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}"},
            json={
                "model": mistral_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 1100
            },
            timeout=60
        )
    except Exception as e:
        return f"[Mistral request error: {e}]"
    if not r.ok:
        return f"[Mistral API error: {r.status_code} {r.text}]"
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
vendor = st.sidebar.selectbox("LLM Vendor", ["OpenAI","Mistral"], index=0)
st.session_state.vendor = vendor

if vendor == "OpenAI":
    # Tools for gpt-4o/4o-mini; Responses for gpt-5-nano
    openai_labels = ["gpt-4o", "gpt-4o-mini", "gpt-5-nano"]
    openai_model_map = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-5-nano": "gpt-5-nano",
    }
    model_label = st.sidebar.selectbox("Model", openai_labels, index=0)
    model_choice = openai_model_map[model_label]
    st.session_state.openai_model_choice = model_choice
else:
    mistral_labels = ["mistral-large-latest", "mistral-medium"]
    mistral_model_map = {
        "mistral-large-latest": "mistral-large-latest",
        "mistral-medium": "mistral-medium",
    }
    model_label = st.sidebar.selectbox("Model", mistral_labels, index=0)
    model_choice = mistral_model_map[model_label]
    st.session_state.mistral_model_choice = model_choice
    st.sidebar.caption(f"Using Mistral model: {model_choice}")

language_ui = st.sidebar.selectbox(
    "Response language",
    ["English","Spanish","French","German","Chinese","Japanese","Russian"], index=0
)
style_choice = st.sidebar.selectbox(
    "Style",
    ["Normal","100 words","2 connected paragraphs","5 bullet points"], index=0
)

# Conversation memory controls (mutually exclusive)
st.sidebar.header("Conversation memory")
mem_mode = st.sidebar.radio(
    "Mode (choose one)",
    ["Buffer: 5 questions", "Conversation summary", "Buffer: 2000 tokens"],
    index=0
)
st.session_state.mem_mode = mem_mode

if mem_mode == "Buffer: 5 questions":
    st.session_state.mem_last_n = st.sidebar.slider("Remember last N questions", 1, 10, MEMORY_MAX_USER_TURNS_DEFAULT)
elif mem_mode == "Buffer: 2000 tokens":
    st.session_state.mem_token_budget = st.sidebar.slider("Token buffer (~tokens)", 500, 8000, MEMORY_TOKEN_BUDGET_DEFAULT, step=100)
# "Conversation summary" has no extra UI; uses SUMMARY_MAX_WORDS constant

# Minimal ingestion controls
min_chunk_allowed = 50
chunk_size_sel = st.sidebar.number_input("Chunk size (chars)", min_value=min_chunk_allowed, max_value=4000, value=max(CHUNK_SIZE_DEFAULT, min_chunk_allowed), step=50)
overlap_sel = st.sidebar.number_input("Overlap (chars)", min_value=0, max_value=1000, value=max(CHUNK_OVERLAP_DEFAULT, 0), step=10)

# ---------------- Load local CSV ----------------
st.header("1) Auto-load CSV and Ingest")

def load_csv_local(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="utf-8", errors="replace")

df = load_csv_local(CSV_FILENAME)

if df is None:
    st.error(f"Local file not found: {CSV_FILENAME}")
    st.stop()

st.success(f"Loaded local file: {CSV_FILENAME} (rows: {len(df)})")
st.dataframe(df.head())

# Detect mode
lower_cols = {c.lower() for c in df.columns}
has_direct_content = {"title","content"}.issubset(lower_cols)
possible_url_cols = [c for c in df.columns if c.lower() in ("url","link","links")]
has_url_mode = (not has_direct_content) and len(possible_url_cols) > 0

st.write("Detected ingestion mode:")
if has_direct_content:
    st.info("Mode A: direct content (title + content).")
elif has_url_mode:
    st.info(f"Mode B: URL ingestion (column: {possible_url_cols[0]}). All unique URLs will be fetched and indexed.")
else:
    st.error("Neither (title, content) nor a URL column found. Nothing to ingest.")
    st.stop()

# Actions
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Wipe Vector DB"):
        try:
            ids = collection.get().get("ids", [])
            if ids:
                collection.delete(ids=ids)
                st.success(f"Deleted {len(ids)} records.")
            else:
                st.info("Vector DB already empty.")
        except Exception as e:
            st.error(f"Error wiping DB: {e}")
with c2:
    reindex_full = st.button("Reindex (full)")
with c3:
    if st.button("Show Chunk Count"):
        try:
            ccount = len(collection.get().get("ids", []))
            st.info(f"Current chunk count in DB: {ccount}")
        except Exception as e:
            st.error(f"Read error: {e}")

def ingest_mode_A(df: pd.DataFrame):
    col_map = {c.lower(): c for c in df.columns}
    title_col = col_map["title"]
    content_col = col_map["content"]
    date_col = col_map.get("date")
    company_col = None
    category_col = None
    for c in df.columns:
        lc = c.lower()
        if "company" in lc and company_col is None:
            company_col = c
        if lc == "category":
            category_col = c

    articles = []
    for i, row in df.iterrows():
        title = str(row.get(title_col,"")).strip()
        content = str(row.get(content_col,"")).strip()
        date_val = str(row.get(date_col,"")).strip() if date_col else ""
        company_name = str(row.get(company_col,"")).strip() if company_col else ""
        category_val = str(row.get(category_col,"")).strip() if category_col else ""
        full_text = f"{title}\n\n{content}"
        articles.append({
            "base_id": f"direct_row{i}",
            "title": title or "Untitled",
            "date": date_val,
            "company_name": company_name,
            "category": category_val,
            "url": "",
            "full_text": full_text,
            "source_type": "direct"
        })
    before = len(collection.get().get("ids", []))
    added = index_articles(articles, chunk_size=chunk_size_sel, overlap=overlap_sel, min_article_len=MIN_ARTICLE_LEN_DEFAULT)
    after = len(collection.get().get("ids", []))
    st.success(f"Ingestion complete. Added chunks: {added}. Total chunks: {after} (was {before}).")

def ingest_mode_B(df: pd.DataFrame):
    url_col = possible_url_cols[0]
    date_col = None
    title_hint_col = None
    company_col = None
    category_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("date","published","pub_date") and date_col is None:
            date_col = c
        if lc in ("document","title","headline","name") and title_hint_col is None:
            title_hint_col = c
        if "company" in lc and company_col is None:
            company_col = c
        if lc == "category":
            category_col = c

    urls = df[url_col].dropna().astype(str).unique().tolist()
    total_urls = len(urls)
    st.write(f"Found unique URLs: {total_urls}")
    if total_urls == 0:
        st.info("No URLs to process.")
        return

    progress = st.progress(0)
    status_box = st.empty()
    articles = []
    success, fail, skipped_empty = 0, 0, 0

    for idx, url in enumerate(urls, 1):
        status_box.info(f"[{idx}/{total_urls}] Fetch: {url}")
        html = fetch_page(url, timeout=int(FETCH_TIMEOUT_DEFAULT))
        if not html:
            fail += 1
            progress.progress(idx/total_urls)
            if SLEEP_BETWEEN_REQUESTS:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        extr = extract_main_text_basic(html)
        page_title = extr.get("title","").strip()
        body = extr.get("body_plain","").strip()

        row = df[df[url_col]==url].iloc[0]
        row_title_hint = str(row.get(title_hint_col,"")).strip() if title_hint_col else ""
        final_title = row_title_hint or page_title or "Untitled Article"

        row_date = str(row.get(date_col,"")).strip() if date_col else ""
        if row_date:
            try:
                row_date_clean = row_date.replace("Z","").split("+")[0]
                dt = datetime.fromisoformat(row_date_clean)
                row_date = dt.isoformat()
            except Exception:
                pass

        company_name = str(row.get(company_col,"")).strip() if company_col else ""
        category_val = str(row.get(category_col,"")).strip() if category_col else ""

        full_text = f"{final_title}\n\n{body}"
        if len(full_text.strip()) == 0:
            skipped_empty += 1
            progress.progress(idx/total_urls)
            if SLEEP_BETWEEN_REQUESTS:
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        articles.append({
            "base_id": f"url_{hash_url(url)}",
            "title": final_title,
            "date": row_date,
            "company_name": company_name,
            "category": category_val,
            "url": url,
            "full_text": full_text,
            "source_type": "fetched"
        })
        success += 1
        progress.progress(idx/total_urls)
        if SLEEP_BETWEEN_REQUESTS:
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    status_box.success(f"Fetch finished: success={success}, failed={fail}, empty={skipped_empty}")
    before = len(collection.get().get("ids", []))
    added_chunks = index_articles(articles, chunk_size=chunk_size_sel, overlap=overlap_sel, min_article_len=MIN_ARTICLE_LEN_DEFAULT)
    after = len(collection.get().get("ids", []))
    st.success(f"Indexing complete. Added chunks: {added_chunks}. Total chunks: {after} (was {before}).")

# Auto-ingest on first run if DB empty
if AUTO_INDEX_ON_START and not reindex_full:
    current_count = len(collection.get().get("ids", []))
    if current_count == 0:
        st.info("Auto-ingest: Vector DB is empty, starting ingestion...")
        if has_direct_content:
            ingest_mode_A(df)
        elif has_url_mode:
            ingest_mode_B(df)

# Full reindex
if reindex_full:
    st.warning("Full reindex: wiping DB first.")
    try:
        ids_all = collection.get().get("ids", [])
        if ids_all:
            collection.delete(ids=ids_all)
            st.info(f"Wiped {len(ids_all)} records.")
    except Exception as e:
        st.error(f"Wipe error: {e}")
    if has_direct_content:
        ingest_mode_A(df)
    elif has_url_mode:
        ingest_mode_B(df)

# Manual run for URL mode
if (not has_direct_content) and has_url_mode and st.button("Process ALL URLs now"):
    ingest_mode_B(df)

# ---------------- View stored chunks ----------------
with st.expander("View stored chunks"):
    try:
        data = collection.get(include=["metadatas","documents"])
        ids = data.get("ids", [])
        mets = data.get("metadatas", [])
        docs = data.get("documents", [])
        total = len(ids)
        st.write(f"Total chunks in DB: {total}")
        if total == 0:
            st.info("No chunks indexed yet.")
        else:
            show_n = st.slider("Show first N", 1, min(50, total), min(20, total))
            for i in range(show_n):
                m = mets[i]
                st.markdown(f"**{i+1}. {m.get('title','(no title)')}** (chunk {m.get('chunk_index')}/{m.get('total_chunks')})")
                st.write(docs[i][:250] + ("..." if len(docs[i])>250 else ""))
                st.caption(f"id={ids[i]} | date={m.get('date','')} | company={m.get('company_name','')} | category={m.get('category','')} | url={m.get('url','')} | source={m.get('source_type','')}")
                st.divider()
    except Exception as e:
        st.error(f"Read error: {e}")

# ---------------- Chat Section ----------------
st.header("2) Chat")

def add_history(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})
    st.session_state.chat_history = st.session_state.chat_history[-120:]

# Render previous
for h in st.session_state.chat_history:
    with st.chat_message(h["role"]):
        st.markdown(h["content"])

user_input = st.chat_input("Ask: 'find the most interesting news' (law-only), 'find news about antitrust' (general), '#2 details', or any question")
if user_input:
    add_history("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            style_txt = style_instruction(style_choice)
            if vendor == "OpenAI":
                system_prompt = (
                    "You are a news assistant.\n"
                    "- If the user asks for 'most interesting news', summarize legal-related items (law-only).\n"
                    "- If the user asks for news about a topic or asks any other question, build a general answer grounded in retrieved context.\n"
                    "- When the user refers to an item by index (e.g., '#2'), use the corresponding article chunks for details.\n"
                    f"Respond in {language_ui}. {style_txt}\n"
                    "Use clear headings and bullets, and include succinct source mentions when available."
                )
                # OpenAI path: tools for tool-capable models; Responses fallback for nano
                answer = openai_function_call(system_prompt, user_input, st.session_state.get("openai_model_choice", "gpt-4o"))
            else:
                # Mistral path: NO tools; app orchestrates, model formats
                answer = mistral_rag_answer(
                    user_query=user_input,
                    language=language_ui,
                    style_txt=style_txt,
                    model=st.session_state.get("mistral_model_choice", "mistral-large-latest")
                )
            st.markdown(answer)
    add_history("assistant", answer)
    # Update running summary if mode requires it
    try:
        update_conversation_summary(user_input, answer)
    except Exception:
        pass

# ---------------- Quick Topic Search (GENERAL, de-duplicated by URL) ----------------
st.header("3) Quick semantic topic search (general, unique news by URL)")
topic_query = st.text_input("Enter a topic (any domain): e.g., AI, oil prices, antitrust, sports")
if st.button("Search Topic (general)"):
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY is required for semantic search.")
    else:
        items = retrieve_unique_news_by_url_general(topic_query, k=5, oversample=40)
        if not items:
            st.info("No results (or empty DB).")
        else:
            for i, it in enumerate(items, 1):
                title = it.get("title","(no title)")
                url = it.get("url","")
                if url:
                    st.markdown(f"**{i}. [{title}]({url})**")
                else:
                    st.markdown(f"**{i}. {title}**")
                st.write(it.get("snippet",""))
                st.caption(f"Date: {it.get('date','')} | Company: {it.get('company_name','')} | Category: {it.get('category','')}")
                st.divider()