# HWs/HW1.py
import streamlit as st
import os
from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

def read_pdf(file):
    text_parts = []
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()

# ---- Main app function ----
def app():
    st.title("📄 HW1")
    st.write(
        "Upload a document below and ask a question about it – GPT will answer! "
        "API key is read automatically from your .env file."
    )

    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Add OPENAI_API_KEY to your .env file.")
        return

    client = OpenAI(api_key=openai_api_key)

    # Choose model
    model = st.selectbox(
        "Choose a model",
        ["gpt-3.5", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"],
        index=1,
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    document = None
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "txt":
            document = uploaded_file.read().decode()
        elif file_extension == "pdf":
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

    # Ask a question
    question = st.text_area(
        "Now ask a question about the document:",
        placeholder="Example: Summarize the document.",
        disabled=not document,
    )

    # Generate answer
    if document and question:
        messages = [
            {"role": "user", "content": f"Here is a document:\n{document}\n\n---\n\n{question}"}
        ]
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            st.subheader("Answer")
            st.write_stream(stream)
        except Exception as e:
            st.error(f"Error while querying the model: {e}")
app()