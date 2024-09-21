import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

openai_api_key = st.secrets["openai"]["api_key"]

OpenAI.api_key = openai_api_key

def load_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.title("📄 Document Search & Question-Answering Bot")
st.write("Upload your PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.write("Extracting text from the PDF...")
    pdf_text = load_pdf(uploaded_file)
    
    st.write("PDF Text Extracted:")
    st.write(pdf_text[:1000])  

    embeddings = OpenAIEmbeddings()

    st.write("Embedding the document...")
    docsearch = FAISS.from_texts([pdf_text], embeddings)

    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=docsearch.as_retriever())

    st.write("Now you can ask questions about the document.")
    user_question = st.text_input("Enter your question:")

    if user_question:
        st.write("Searching for the answer...")
        answer = qa_chain.run(user_question)
        
        st.write("Answer:")
        st.write(answer)

else:
    st.write("Please upload a PDF file to start.")
