# =============================================================
# HW7 Auto-Ingest News RAG Bot (All URLs, English UI)
# Single-file Streamlit app
# =============================================================
# Features:
# - Automatically loads local CSV: Example_news_info_for_testing.csv
# - Mode A (direct): if CSV has columns title + content, indexes them directly
# - Mode B (URLs): if no content but has URL column, fetches ALL pages, extracts text, chunks, embeds, stores in Chroma
# - Caches HTML in ./page_cache
# - "Only new URLs" option (skip already indexed URLs)
# - Reindex (full), Wipe DB, show chunk list, quick semantic search
# - OpenAI Function Calling tools:
#     * find_relevant_news(topic)
#     * rank_interesting_news(limit)
# - Mistral vendor (context-only RAG), optional
#
# Requirements:
# pip install streamlit chromadb openai beautifulsoup4 requests pandas
#
# Secrets (.streamlit/secrets.toml) or ENV:
# OPENAI_API_KEY="sk-..."
# MISTRAL_API_KEY="..."   # optional
#
# Run:
# streamlit run hw7_app.py
# =============================================================

# Optional sqlite fallback (some environments)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

import os
import json
import time
import re
import hashlib
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import chromadb
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- Configuration ----------------
CSV_FILENAME = "Example_news_info_for_testing.csv"   # Local CSV expected in the same directory
CHROMA_PATH = "./news_chroma_db"
COLLECTION_NAME = "NewsCollection"
EMBEDDING_MODEL = "text-embedding-3-small"
PAGE_CACHE_DIR = "./page_cache"

CHUNK_SIZE_DEFAULT = 900
CHUNK_OVERLAP_DEFAULT = 150
MIN_ARTICLE_LEN_DEFAULT = 300     # min length of combined text (title+body) to index
FETCH_TIMEOUT_DEFAULT = 20
SLEEP_BETWEEN_REQUESTS = 0.15     # polite delay between page fetches (seconds)
USER_AGENT = "Mozilla/5.0 (compatible; HW7AllURLsBot/1.0; +https://example.com/bot)"

AUTO_INDEX_ON_START = True  # auto-ingest if DB is empty

st.set_page_config(page_title="HW7 Auto News RAG (All URLs)", layout="wide")
st.title("📰 HW7 Auto News RAG Bot — All URLs")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY", os.environ.get("MISTRAL_API_KEY", ""))

if not OPENAI_KEY:
    st.warning("OPENAI_API_KEY not found — embeddings and OpenAI chat will not work.")
if not MISTRAL_KEY:
    st.info("MISTRAL_API_KEY not set — Mistral path is optional.")

os.makedirs(PAGE_CACHE_DIR, exist_ok=True)

# ---------------- Chroma Init ----------------
@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)

collection = get_chroma_collection()

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

# ---------------- Helpers: Hash & Cache ----------------
def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]

def cache_path_for(url: str) -> str:
    return os.path.join(PAGE_CACHE_DIR, f"{hash_url(url)}.json")

def fetch_page(url: str, timeout: int) -> Optional[str]:
    """
    Fetch page with simple disk cache.
    Returns raw HTML or None if failed.
    """
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
    """
    Very basic extractor:
      - remove script/style/noscript/header/footer/svg
      - collect h1/h2/h3/p
    Returns: { title, body_plain }
    """
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

def get_already_indexed_urls() -> set:
    """
    Read metadata from collection and collect URLs that are already indexed.
    """
    try:
        data = collection.get(include=["metadatas"])
        mets = data.get("metadatas", [])
        return {m.get("url","") for m in mets if m.get("url")}
    except Exception:
        return set()

def index_articles(
    articles: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_article_len: int
) -> int:
    """
    Articles list element format:
    {
      "base_id": str,
      "title": str,
      "date": str,
      "company_name": str,
      "url": str,
      "full_text": str,
      "source_type": "direct" | "fetched"
    }
    """
    try:
        existing_ids = set(collection.get().get("ids", []))
    except Exception:
        existing_ids = set()
    added = 0
    for art in articles:
        full_text = art["full_text"]
        if len(full_text) < min_article_len:
            continue
        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        for ci, ch in enumerate(chunks):
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

# ---------------- Retrieval ----------------
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
            "url": meta.get("url",""),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "text": doc
        })
    return out

# ---------------- Tools (Function Calling) ----------------
def style_instruction(style: str) -> str:
    return {
        "Normal": "",
        "100 words": "Answer in about 100 words.",
        "2 connected paragraphs": "Answer in exactly two connected paragraphs.",
        "5 bullet points": "Provide exactly five concise bullet points."
    }.get(style, "")

def tool_find_relevant_news(topic: str) -> str:
    found = retrieve_top_k(topic, 5)
    payload = []
    for f in found:
        payload.append({
            "title": f["title"],
            "date": f["date"],
            "company_name": f["company_name"],
            "url": f["url"],
            "chunk_index": f["chunk_index"],
            "total_chunks": f["total_chunks"],
            "snippet": f["text"][:300] + ("..." if len(f["text"])>300 else "")
        })
    return json.dumps(payload, ensure_ascii=False, indent=2)

def tool_rank_interesting_news(limit: int = 5) -> str:
    data = collection.get(include=["metadatas","documents"])
    mets = data.get("metadatas", [])
    docs = data.get("documents", [])
    now = datetime.utcnow()
    scored = []
    for meta, doc in zip(mets, docs):
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
        first_chunk_bonus = 1.0 if meta.get("chunk_index",0)==0 else 0.6
        score = 0.5*recency + 0.3*length_score + 0.2*first_chunk_bonus
        scored.append((score, meta, doc))
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:max(1,min(10,limit))]
    out = []
    for i,(sc,m,dtext) in enumerate(ranked,1):
        out.append({
            "rank": i,
            "title": m.get("title",""),
            "date": m.get("date",""),
            "company_name": m.get("company_name",""),
            "url": m.get("url",""),
            "score": round(sc,4),
            "snippet": dtext[:250] + ("..." if len(dtext)>250 else "")
        })
    return json.dumps(out, ensure_ascii=False, indent=2)

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "find_relevant_news",
            "description": "Find top 5 news chunks about a specific topic (semantic retrieval).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic or keyword"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_interesting_news",
            "description": "Return ranked list of globally most interesting news chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "How many items (1-10)", "default": 5}
                }
            }
        }
    }
]

def dispatch_tool_call(name: str, args: Dict[str, Any]) -> str:
    if name == "find_relevant_news":
        return tool_find_relevant_news(args.get("topic",""))
    if name == "rank_interesting_news":
        return tool_rank_interesting_news(args.get("limit",5))
    return "Tool not recognized"

# ---------------- OpenAI Function Calling (FIXED follow-up) ----------------
def openai_function_call(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Fixed: follow-up includes the assistant message with tool_calls
    to satisfy the OpenAI API requirement.
    """
    if not OPENAI_KEY:
        return "OPENAI_API_KEY is missing."
    client = get_openai_client()

    # 1) First call (allow tool use)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        first = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_OPENAI,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=900
        )
    except Exception as e:
        return f"[OpenAI API error: {e}]"

    first_msg = first.choices[0].message

    # If no tool was called, return the direct answer
    if not getattr(first_msg, "tool_calls", None):
        return first_msg.content

    # 2) Execute the tool locally
    tool_call = first_msg.tool_calls[0]
    import json as pyjson
    try:
        tool_args = pyjson.loads(tool_call.function.arguments or "{}")
    except Exception:
        tool_args = {}
    tool_output = dispatch_tool_call(tool_call.function.name, tool_args)

    # 3) Follow-up call: MUST include the assistant message with tool_calls before the tool message
    follow_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": tool_output,
        },
    ]
    try:
        second = client.chat.completions.create(
            model=model,
            messages=follow_messages,
            temperature=0.7,
            max_tokens=900
        )
    except Exception as e:
        return f"[OpenAI follow-up error: {e}]"
    return second.choices[0].message.content

# ---------------- Mistral RAG (no function calling) ----------------
def mistral_rag_answer(user_query: str, language: str, style_txt: str, model: str) -> str:
    if not MISTRAL_KEY:
        return "Mistral key is missing."
    if not OPENAI_KEY:
        return "OpenAI key is required for semantic retrieval."
    retrieved = retrieve_top_k(user_query, 5)
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[Title: {r['title']} | Date: {r['date']} | Chunk {r['chunk_index']}/{r['total_chunks']}] {r['text'][:600]}"
        )
    context_text = "\n\n".join(context_blocks) if context_blocks else "NO_CONTEXT"
    prompt = (
        f"You are a legal news assistant. Respond in {language}. {style_txt}\n"
        f"Use ONLY the context if relevant. If no context, say you lack data.\n\n"
        f"Context:\n{context_text}\n\nUser question: {user_query}"
    )
    try:
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}"},
            json={"model": model,"messages":[{"role":"user","content":prompt}],"temperature":0.7,"max_tokens":900},
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
model_choice = st.sidebar.selectbox("Model", ["gpt-4o","gpt-4-turbo","mistral-large-latest"], index=0)
language_ui = st.sidebar.selectbox(
    "Response language",
    ["English","Spanish","French","German","Chinese","Japanese","Russian"], index=0
)
style_choice = st.sidebar.selectbox(
    "Style",
    ["Normal","100 words","2 connected paragraphs","5 bullet points"], index=0
)
chunk_size_sel = st.sidebar.number_input("Chunk size (chars)", 300, 4000, CHUNK_SIZE_DEFAULT, 50)
overlap_sel = st.sidebar.number_input("Overlap (chars)", 0, 1000, CHUNK_OVERLAP_DEFAULT, 10)
min_article_len_sel = st.sidebar.number_input("Min article length (chars)", 100, 8000, MIN_ARTICLE_LEN_DEFAULT, 50)
fetch_timeout_sel = st.sidebar.number_input("Fetch timeout (sec)", 5, 120, FETCH_TIMEOUT_DEFAULT, 1)
sleep_between_sel = st.sidebar.number_input("Sleep between fetches (sec)", 0.0, 5.0, SLEEP_BETWEEN_REQUESTS, 0.05)
only_new = st.sidebar.checkbox("Index only new URLs (recommended)", True)

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
has_direct_content = {"title","content"}.issubset({c.lower() for c in df.columns})
possible_url_cols = [c for c in df.columns if c.lower() in ("url","link","links")]
has_url_mode = (not has_direct_content) and len(possible_url_cols) > 0

st.write("Detected ingestion mode:")
if has_direct_content:
    st.info("Mode A: direct content (title + content).")
elif has_url_mode:
    st.info(f"Mode B: URL ingestion (column: {possible_url_cols[0]}).")
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

# Ingestion routines
def ingest_mode_A(df: pd.DataFrame):
    col_map = {c.lower(): c for c in df.columns}
    title_col = col_map["title"]
    content_col = col_map["content"]
    date_col = col_map.get("date")
    company_col = None
    for c in df.columns:
        if "company" in c.lower():
            company_col = c
            break

    articles = []
    for i, row in df.iterrows():
        title = str(row.get(title_col,"")).strip()
        content = str(row.get(content_col,"")).strip()
        date_val = ""
        if date_col:
            date_val = str(row.get(date_col,"")).strip()
        company_name = ""
        if company_col:
            company_name = str(row.get(company_col,"")).strip()
        full_text = f"{title}\n\n{content}"
        articles.append({
            "base_id": f"direct_row{i}",
            "title": title or "Untitled",
            "date": date_val,
            "company_name": company_name,
            "url": "",
            "full_text": full_text,
            "source_type": "direct"
        })
    before = len(collection.get().get("ids", []))
    added = index_articles(
        articles,
        chunk_size=chunk_size_sel,
        overlap=overlap_sel,
        min_article_len=min_article_len_sel
    )
    after = len(collection.get().get("ids", []))
    st.success(f"Ingestion complete. Added chunks: {added}. Total chunks: {after} (was {before}).")

def ingest_mode_B(df: pd.DataFrame):
    url_col = possible_url_cols[0]
    date_col = None
    doc_col = None
    company_col = None

    for c in df.columns:
        lc = c.lower()
        if lc in ("date","published","pub_date") and date_col is None:
            date_col = c
        if lc in ("document","title","headline","name") and doc_col is None:
            doc_col = c
        if "company" in lc and company_col is None:
            company_col = c

    urls = df[url_col].dropna().astype(str).unique().tolist()
    total_urls = len(urls)
    st.write(f"Found unique URLs: {total_urls}")

    existing_urls = get_already_indexed_urls() if only_new else set()
    if only_new:
        urls = [u for u in urls if u not in existing_urls]
        st.write(f"New URLs to process: {len(urls)} (already indexed: {len(existing_urls)})")

    if not urls:
        st.info("No new URLs to process.")
        return

    progress = st.progress(0)
    status_box = st.empty()
    articles = []
    success, fail, skipped_short = 0, 0, 0

    for idx, url in enumerate(urls, 1):
        status_box.info(f"[{idx}/{len(urls)}] Fetch: {url}")
        html = fetch_page(url, timeout=int(fetch_timeout_sel))
        if not html:
            fail += 1
            progress.progress(idx/len(urls))
            if sleep_between_sel:
                time.sleep(sleep_between_sel)
            continue
        extr = extract_main_text_basic(html)
        page_title = extr.get("title","").strip()
        body = extr.get("body_plain","").strip()

        row = df[df[url_col]==url].iloc[0]

        row_title = ""
        if doc_col:
            row_title = str(row.get(doc_col,"")).strip()
        final_title = row_title or page_title or "Untitled Article"

        row_date = ""
        if date_col:
            row_date = str(row.get(date_col,"")).strip()
            if row_date:
                try:
                    row_date_clean = row_date.replace("Z","").split("+")[0]
                    dt = datetime.fromisoformat(row_date_clean)
                    row_date = dt.isoformat()
                except Exception:
                    pass

        company_name = ""
        if company_col:
            company_name = str(row.get(company_col,"")).strip()

        full_text = f"{final_title}\n\n{body}"
        if len(full_text) < min_article_len_sel:
            skipped_short += 1
            progress.progress(idx/len(urls))
            if sleep_between_sel:
                time.sleep(sleep_between_sel)
            continue

        articles.append({
            "base_id": f"url_{hash_url(url)}",
            "title": final_title,
            "date": row_date,
            "company_name": company_name,
            "url": url,
            "full_text": full_text,
            "source_type": "fetched"
        })
        success += 1
        progress.progress(idx/len(urls))
        if sleep_between_sel:
            time.sleep(sleep_between_sel)

    status_box.success(f"Fetch complete: success={success}, fail={fail}, skipped_short={skipped_short}")
    before = len(collection.get().get("ids", []))
    added_chunks = index_articles(
        articles,
        chunk_size=chunk_size_sel,
        overlap=overlap_sel,
        min_article_len=min_article_len_sel
    )
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

# Full reindex button
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

# Manual run for URL mode (process all or only new)
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
                st.markdown(
                    f"**{i+1}. {m.get('title','(no title)')}** "
                    f"(chunk {m.get('chunk_index')}/{m.get('total_chunks')})"
                )
                st.write(docs[i][:250] + ("..." if len(docs[i])>250 else ""))
                st.caption(
                    f"id={ids[i]} | date={m.get('date','')} | company={m.get('company_name','')} "
                    f"| url={m.get('url','')} | source={m.get('source_type','')}"
                )
                st.divider()
    except Exception as e:
        st.error(f"Read error: {e}")

# ---------------- Chat Section ----------------
st.header("2) Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_history(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})
    st.session_state.chat_history = st.session_state.chat_history[-30:]

for h in st.session_state.chat_history:
    with st.chat_message(h["role"]):
        st.markdown(h["content"])

user_input = st.chat_input("Ask something about the news (e.g., 'find the most interesting news')")
if user_input:
    add_history("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            style_txt = style_instruction(style_choice)
            if vendor == "OpenAI":
                system_prompt = f"You are a legal news assistant. Respond in {language_ui}. {style_txt}"
                model_for_call = model_choice if model_choice.startswith("gpt-") else "gpt-4o"
                answer = openai_function_call(system_prompt, user_input, model_for_call)
            else:
                answer = mistral_rag_answer(
                    user_query=user_input,
                    language=language_ui,
                    style_txt=style_txt,
                    model=model_choice if model_choice.startswith("mistral") else "mistral-large-latest"
                )
            st.markdown(answer)
    add_history("assistant", answer)

# ---------------- Quick Topic Search ----------------
st.header("3) Quick semantic topic search")
topic_query = st.text_input("Enter a topic keyword (e.g., antitrust, compliance, mergers)")
if st.button("Search Topic"):
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY is required for semantic search.")
    else:
        res = retrieve_top_k(topic_query, 5)
        if not res:
            st.info("No results (or empty DB).")
        else:
            for i, r in enumerate(res, 1):
                st.markdown(f"**{i}. {r['title']}** (chunk {r['chunk_index']}/{r['total_chunks']})")
                st.write(r['text'][:400] + ("..." if len(r['text'])>400 else ""))
                st.caption(
                    f"Date: {r['date']} | Company: {r['company_name']} | URL: {r['url']}"
                )
                st.divider()

st.markdown("---")
st.caption("HW7 Auto-Ingest All URLs • 2025 (English UI)")