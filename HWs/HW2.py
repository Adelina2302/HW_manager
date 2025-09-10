import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import cohere

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        lines = [line.strip() for line in soup.get_text().splitlines()]
        return "\n".join([line for line in lines if line])
    except Exception as e:
        st.error(f"Error reading {url}: {e}")
        return None

def style_instruction(style):
    if style == "100 words":
        return "Produce a single coherent summary of about 100 words."
    if style == "2 connected paragraphs":
        return "Produce a concise summary as exactly two connected paragraphs."
    if style == "5 bullet points":
        return "Produce exactly five concise bullet points."
    return "Produce a concise summary."

def build_prompt(language, style, text):
    return f"""
You are a careful academic summarizer.
Write the output in {language}.
{style_instruction(style)}

SOURCE TEXT START
{text}
SOURCE TEXT END
""".strip()

def openai_summarize(use_advanced, language, style, text):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("No OpenAI API key found!")
        return None
    client = OpenAI(api_key=api_key)
    prompt = build_prompt(language, style, text)
    model = "gpt-4o" if use_advanced else "gpt-3.5-turbo"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

def mistral_summarize(use_advanced, language, style, text):
    api_key = st.secrets.get("MISTRAL_API_KEY", "")
    if not api_key:
        st.error("No Mistral API key found!")
        return None
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1"
    )
    prompt = build_prompt(language, style, text)
    model = "mistral-large-latest" if use_advanced else "mistral-small-latest"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful academic summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Mistral error: {e}")
        return None

def cohere_summarize(use_advanced, language, style, text):
    api_key = st.secrets.get("COHERE_API_KEY", "")
    if not api_key:
        st.error("No Cohere API key found!")
        return None
    client = cohere.Client(api_key)
    prompt = build_prompt(language, style, text)
    model = "command-nightly" if use_advanced else "command"
    # Cohere chat endpoint (recommended for new API keys)
    try:
        response = client.chat(
            model=model,
            message=prompt,
            temperature=0.2,
            max_tokens=500
        )
        return response.text.strip()
    except Exception as e:
        # Если chat не работает, пробуем через generate
        try:
            resp = client.generate(prompt=prompt, max_tokens=500, temperature=0.2)
            return resp.generations[0].text.strip()
        except Exception as e2:
            st.error(f"Cohere error: {e2}")
            return None

def app():
    st.title("🧠 Multi-LLM Summarizer")
    st.markdown("Enter a URL below and select an LLM (and advanced model, if desired) to generate a summary.")

    url = st.text_input("Enter a URL to summarize:")
    st.sidebar.header("Summary Controls")
    language = st.sidebar.selectbox(
        "Output language:",
        ["English", "French", "Spanish", "German", "Russian", "Chinese"]
    )
    summary_style = st.sidebar.selectbox(
        "Summary style:",
        ["100 words", "2 connected paragraphs", "5 bullet points"]
    )

    llm_choice = st.sidebar.selectbox(
        "Select LLM:",
        ["OpenAI", "Mistral", "Cohere"]
    )
    use_advanced = st.sidebar.checkbox("Use advanced model", value=False)

    llm_map = {
        "OpenAI": openai_summarize,
        "Mistral": mistral_summarize,
        "Cohere": cohere_summarize
    }

    if url:
        with st.spinner("Fetching content..."):
            text = read_url_content(url)
        if not text:
            st.error("No readable text found at this URL.")
            return

        llm_func = llm_map.get(llm_choice)
        out = None
        if llm_func:
            with st.spinner(f"Summarizing with {llm_choice}..."):
                out = llm_func(use_advanced, language, summary_style, text)

        if out:
            st.subheader(f"Summary in {language} ({llm_choice})")
            st.write(out)
            st.download_button(
                "Download summary as .txt",
                data=out.encode("utf-8"),
                file_name="url_summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    app()