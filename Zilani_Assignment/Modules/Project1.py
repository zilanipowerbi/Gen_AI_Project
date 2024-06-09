# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:22:34 2024

@author: shaikzil
"""

import streamlit as st
import streamlit.web.bootstrap
import streamlit.web.cli
import os
import tiktoken
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain


# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarize_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()

    # Store the documents in the vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(collection_name="pdf_documents", embedding_function=embeddings)
    vector_store.add_documents(docs)

    llm = OpenAI(temperature=1)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

def main():
    st.set_page_config(page_title="GEN_AI Assignment - Zilan Basha Shaik - PDF Summarizer", layout="wide")
    st.title("PDF Summarizer")

    # Create a file uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Summarize the PDF file
        summary = summarize_pdf("uploaded_file.pdf")

        # Display the summary
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please upload a PDF file to get started.")

if __name__ == "__main__":
    main()