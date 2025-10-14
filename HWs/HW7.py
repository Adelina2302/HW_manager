__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import json
from io import StringIO

import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import chromadb
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="HW7: News Info Bot", layout="wide")
st.title("📰 HW7: News Info Bot (RAG + Tools + Model Comparison)")

DEFAULT_CSV = "Example_news_info_for_testing.csv"
CHROMA_PATH = "./ChromaDB_news"
EMBED_CACHE_PATH = "./embeddings_cache.json"

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    st.error("❌ OPENAI_API_KEY not set in Streamlit Secrets.")
    st.stop()

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_openai_client():
    return OpenAI(api_key=OPENAI_KEY)

@st.cache_resource(show_spinner=False)
def get_chromadb_collection():
    client = chromadb.PersistentClient(CHROMA_PATH)
    coll = client.get_or_create_collection("NewsCollection")
    return coll

def load_csv_if_exists(path=DEFAULT_CSV):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return pd.read_csv(path, encoding="utf-8", errors="replace")
    return None

def safe_save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def get_embedding(client, text, model="text-embedding-3-small"):
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

# -----------------------------
# VECTOR DB / EMBEDDING MANAGEMENT
# -----------------------------
openai_client = get_openai_client()
collection = get_chromadb_collection()

emb_cache = safe_load_json(EMBED_CACHE_PATH)

def add_news_to_collection(df, client, collection):
    try:
        existing = collection.get()["ids"]
        existing_ids = set(existing)
    except Exception:
        existing_ids = set()

    added = 0
    for _, row in df.iterrows():
        sid = str(row["id"])
        if sid in existing_ids:
            continue
        content = f"{row.get('title','')}\n\n{row.get('content','')}"
        emb = get_embedding(client, content)
        collection.add(
            ids=[sid],
            documents=[content],
            embeddings=[emb],
            metadatas=[{"title": row.get("title", ""), "date": str(row.get("date", "")), "category": row.get("category", "")}]
        )
        emb_cache[sid] = emb
        added += 1

    if added > 0:
        safe_save_json(emb_cache, EMBED_CACHE_PATH)
    return added

def fetch_relevant_news_by_chroma(query, collection, client, n=5):
    q_emb = get_embedding(client, query)
    results = collection.query(query_embeddings=[q_emb], n_results=n)
    news_items = []
    docs = results.get("documents", [[]])[0]
    mets = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    for i, doc in enumerate(docs):
        meta = mets[i] if i < len(mets) else {}
        nid = ids[i] if i < len(ids) else None
        news_items.append({
            "id": nid,
            "title": meta.get("title", "") if meta else "",
            "date": meta.get("date", "") if meta else "",
            "category": meta.get("category", "") if meta else "",
            "content": doc
        })
    return news_items

# -----------------------------
# TOOLS
# -----------------------------
def rank_news_from_df(df, query, client, topk=5):
    q_emb = get_embedding(client, query)
    sims = []
    ids_to_save = {}
    for _, row in df.iterrows():
        sid = str(row["id"])
        content = f"{row.get('title','')}\n\n{row.get('content','')}"
        if sid in emb_cache:
            doc_emb = np.array(emb_cache[sid], dtype=float)
        else:
            doc_emb = np.array(get_embedding(client, content), dtype=float)
            ids_to_save[sid] = doc_emb.tolist()
        sim = float(cosine_similarity([q_emb], [doc_emb])[0][0])
        sims.append(sim)
    if ids_to_save:
        emb_cache.update({k: v for k, v in ids_to_save.items()})
        safe_save_json(emb_cache, EMBED_CACHE_PATH)
    df2 = df.copy()
    df2["similarity"] = sims
    return df2.sort_values("similarity", ascending=False).head(topk)

def find_news_by_topic(df, topic, topk=5):
    mask = df["title"].str.contains(topic, case=False, na=False) | df["content"].str.contains(topic, case=False, na=False)
    return df[mask].head(topk)

# -----------------------------
# LLM CALLS
# -----------------------------
def call_openai_once(model, messages, max_tokens=512, temperature=0.7):
    client = OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return getattr(resp, "output_text", "")

def build_rag_prompt(query, retrieved_news):
    if not retrieved_news:
        return f"You are a helpful news reporting assistant. The user asked: {query}. Answer concisely and avoid fabrications."
    ctx = "\n\n".join([f"{i+1}. {n.get('title','')[:150]} ({n.get('date','')}, {n.get('category','')}): {n.get('content','')[:600]}" for i, n in enumerate(retrieved_news)])
    system = (
        "You are a precise news reporting assistant. When using provided news, start with 'Based on the provided news:'. "
        "Indicate the source titles used and keep the answer concise.\n\n"
        f"Context:\n{ctx}\n\nUser question: {query}\n"
    )
    return system

def compare_models_on_query(query, retrieved_news, small_model="gpt-5-nano", big_model="gpt-4o"):
    rag_prompt = build_rag_prompt(query, retrieved_news)
    messages = [{"role": "system", "content": rag_prompt}, {"role": "user", "content": query}]

    t0 = time.time()
    small_ans = call_openai_once(small_model, messages)
    t_small = time.time() - t0

    t0 = time.time()
    big_ans = call_openai_once(big_model, messages)
    t_big = time.time() - t0

    return {
        "small": {"model": small_model, "answer": small_ans, "latency_s": t_small},
        "big": {"model": big_model, "answer": big_ans, "latency_s": t_big}
    }

# -----------------------------
# UI
# -----------------------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Chatbot (RAG)", "Compare models", "Functions demo"])
language = st.sidebar.selectbox("Output language", ["English", "Russian"], index=0)
openai_models = ["gpt-5-nano", "gpt-4o", "gpt-4-turbo"]
selected_openai_model = st.sidebar.selectbox("OpenAI model", openai_models, index=0)
st.sidebar.markdown("Default CSV: Example_news_info_for_testing.csv (auto-load if available)")

st.header("1) Upload or Load CSV")
uploaded_file = st.file_uploader("Upload CSV (id,title,content,date,category) or auto-load Example_news_info_for_testing.csv", type=["csv"])

if uploaded_file is not None:
    try:
        news_df = pd.read_csv(uploaded_file)
    except Exception:
        news_df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")
    st.success("CSV loaded from upload.")
else:
    auto_df = load_csv_if_exists(DEFAULT_CSV)
    if auto_df is not None:
        news_df = auto_df
        st.info(f"CSV auto-loaded from {DEFAULT_CSV}")
    else:
        st.warning("No CSV loaded. Upload a file or place Example_news_info_for_testing.csv next to this script.")
        st.stop()

required_cols = {"id", "title", "content", "date", "category"}
if not required_cols.issubset(set(news_df.columns)):
    st.error(f"CSV must have columns: {required_cols}. Current columns: {list(news_df.columns)}")
    st.stop()

with st.expander("Sample of loaded CSV"):
    st.dataframe(news_df.head())

if st.button("Add CSV to Vector DB"):
    added_count = add_news_to_collection(news_df, openai_client, collection)
    if added_count > 0:
        st.success(f"Added {added_count} entries to ChromaDB.")
    else:
        st.info("No new entries added (all already present).")

if st.checkbox("Show reset option"):
    if st.button("Clear vector DB"):
        try:
            ids_all = collection.get()["ids"]
            if ids_all:
                collection.delete(ids=ids_all)
            if os.path.exists(EMBED_CACHE_PATH):
                os.remove(EMBED_CACHE_PATH)
            emb_cache.clear()
            st.success("Vector DB and embedding cache cleared.")
        except Exception as e:
            st.error(f"Error clearing DB: {e}")

st.markdown("---")

# -----------------------------
# MAIN MODES
# -----------------------------
if mode == "Chatbot (RAG)":
    st.header("2) Chatbot (RAG)")
    user_query = st.text_input("Enter your question (e.g., 'Find the most interesting news about AI')")
    if st.button("Ask"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            q_low = user_query.lower()
            if "interesting" in q_low or "top news" in q_low:
                st.subheader("Top-5 most relevant news:")
                ranked = rank_news_from_df(news_df, user_query, openai_client, topk=5)
                for i, row in ranked.reset_index(drop=True).iterrows():
                    st.markdown(f"**{i+1}. {row['title']}** — score: {row['similarity']:.4f}")
                    st.write(row['content'][:500] + ("..." if len(row['content']) > 500 else ""))
                    st.markdown("---")
            elif "about" in q_low:
                try:
                    topic = user_query.split("about", 1)[1].strip()
                except Exception:
                    topic = ""
                if topic:
                    found = find_news_by_topic(news_df, topic, topk=10)
                    if found.empty:
                        st.info(f"No news found for topic '{topic}'.")
                    else:
                        st.subheader(f"News about '{topic}':")
                        for _, row in found.iterrows():
                            st.markdown(f"**{row['title']}** — {row['date']} ({row['category']})")
                            st.write(row['content'][:500] + ("..." if len(row['content']) > 500 else ""))
                            st.markdown("---")
                else:
                    st.info("Could not detect topic after 'about'. Try 'Find news about <topic>'.")
            else:
                retrieved = fetch_relevant_news_by_chroma(user_query, collection, openai_client, n=5)
                rag_prompt = build_rag_prompt(user_query, retrieved)
                messages = [{"role": "system", "content": rag_prompt}, {"role": "user", "content": user_query}]
                with st.spinner("Calling model..."):
                    try:
                        model_to_call = selected_openai_model
                        answer = call_openai_once(model_to_call, messages)
                        st.markdown("### Model Answer:")
                        st.write(answer)
                        if retrieved:
                            with st.expander("Documents used as context"):
                                for i, doc in enumerate(retrieved):
                                    st.markdown(f"**Doc {i+1}:** {doc.get('title','')} — {doc.get('date','')} ({doc.get('category','')})")
                                    st.write(doc.get("content","")[:800] + ("..." if len(doc.get("content","")) > 800 else ""))
                    except Exception as e:
                        st.error(f"LLM error: {e}")

elif mode == "Compare models":
    st.header("⚖️ Compare models: gpt-5-nano vs gpt-4o")
    cmp_query = st.text_input("Enter query for comparison (RAG will be used)")
    if st.button("Compare now"):
        if not cmp_query.strip():
            st.warning("Enter a query for comparison.")
        else:
            retrieved = fetch_relevant_news_by_chroma(cmp_query, collection, openai_client, n=5)
            with st.spinner("Comparing models..."):
                try:
                    res = compare_models_on_query(cmp_query, retrieved)
                    st.subheader(f"{res['small']['model']} ({res['small']['latency_s']:.2f}s):")
                    st.write(res['small']['answer'])
                    st.markdown("---")
                    st.subheader(f"{res['big']['model']} ({res['big']['latency_s']:.2f}s):")
                    st.write(res['big']['answer'])
                    ranked_df = rank_news_from_df(news_df, cmp_query, openai_client, topk=5)
                    st.markdown("**Top-5 by semantic rank:**")
                    for i, r in enumerate(ranked_df.itertuples(), 1):
                        st.write(f"{i}. {r.title} (id={r.id}) — score={r.similarity:.3f}")
                except Exception as e:
                    st.error(f"Comparison error: {e}")

elif mode == "Functions demo":
    st.header("🧩 Functions demo: rank_news / find_news_by_topic")
    demo_q = st.text_input("Query for demo (e.g., 'ai', 'health', 'economy')")
    if st.button("Run functions"):
        if not demo_q.strip():
            st.warning("Enter a query.")
        else:
            st.subheader("rank_news (top-5 semantic):")
            r = rank_news_from_df(news_df, demo_q, openai_client, topk=5)
            st.dataframe(r[["id", "title", "similarity"]])
            st.subheader("find_news_by_topic (substring search):")
            f = find_news_by_topic(news_df, demo_q, topk=5)
            st.dataframe(f[["id", "title", "date", "category"]])

st.markdown("---")
st.markdown(
    """
    **Instructions**
    - Place `Example_news_info_for_testing.csv` in the same directory or upload it.
    - Ensure `.streamlit/secrets.toml` includes `OPENAI_API_KEY`.
    - When adding to the vector DB, embeddings will be created and cached.
    - Architecture: RAG (ChromaDB) + Tools (rank/find) + LLM layer (gpt-5-nano vs gpt-4o) + Evaluation (semantic overlap).
    """
)
