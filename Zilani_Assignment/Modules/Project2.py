
import streamlit as st
import streamlit.web.bootstrap
import streamlit.web.cli
import os
import tiktoken

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain


# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarize_and_qa_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()

    # Store the documents in the vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(collection_name="pdf_documents", embedding_function=embeddings)
    vector_store.add_documents(docs)

    llm = OpenAI(temperature=1)
    summarize_chain = load_qa_chain(llm, chain_type="stuff")
    summary = summarize_chain.run(docs)

    # Set up the question-answering chain
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    return summary, qa_chain, vector_store

def main():
    st.set_page_config(page_title="GEN_AI Assignment - Zilan Basha Shaik - PDF Summarizer", layout="wide")
    st.title("PDF Summarizer and Question Answering")
# 
    # Create a file uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Summarize and set up the question-answering chain
        summary, qa_chain, vector_store = summarize_and_qa_pdf("uploaded_file.pdf")

        # Display the summary
        st.subheader("Summary")
        st.write(summary)

        # Add a user prompt for questions
        user_question = st.text_input("Ask a question about the PDF content:")

        if user_question and len(user_question.strip()) > 0:
            print(f"User question: {user_question}")
            # Use the question-answering chain to generate a response
            relevant_docs = list(vector_store.similarity_search(user_question, k=3))
            response = qa_chain.run(input_documents=relevant_docs, question=user_question)
            st.subheader("Response:")
            st.write(response)
        else:
            st.warning("Please enter a question to get started.")
    else:
        st.warning("Please upload a PDF file to get started.")

if __name__ == "__main__":
    main()