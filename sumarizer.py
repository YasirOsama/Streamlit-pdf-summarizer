import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# Function to split large text into chunks (optional for large PDFs)
def get_text_chunks(text, chunk_size=10000, overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to summarize text using Gemini
def summarize_text(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Summarize the following document:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="📚 PDF Summarizer", layout="wide")
st.title("📄 PDF Summarizer using Gemini")

# Upload PDFs
pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

if st.button("Summarize PDFs"):
    if pdf_docs:
        with st.spinner("Reading and summarizing..."):
            full_text = get_pdf_text(pdf_docs)

            # Optional: chunk if large
            if len(full_text) > 10000:
                chunks = get_text_chunks(full_text)
                combined_summary = ""
                for i, chunk in enumerate(chunks):
                    st.info(f"Summarizing chunk {i+1} of {len(chunks)}...")
                    summary = summarize_text(chunk)
                    combined_summary += f"\n\n## Summary of Chunk {i+1}\n{summary}"
                final_summary = combined_summary
            else:
                final_summary = summarize_text(full_text)

            st.subheader("📝 Summary:")
            st.markdown(final_summary)
    else:
        st.warning("Please upload at least one PDF file.")
