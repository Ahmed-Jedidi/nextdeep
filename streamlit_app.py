import streamlit as st
import streamlit_color

from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings  # <-- This needs to be replaced
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import langchain_community
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import Ollama  # <-- Change this to Ollama
from langchain_community.llms import Ollama
# from langchain_openai import OpenAIEmbeddings

import pandas as pd

import os
streamlit_color.main()

os.environ["OLLAMA_API_KEY"] = "ollama"  # <-- Ollama key for authentication if required
os.environ["OPENAI_API_KEY"] = "ollama"  # <-- Ollama key for authentication if required

st.title("Upload Data Files")

# Allow users to upload the first data file
cv1 = st.file_uploader("Upload CV 1", type=["pdf"])

# Allow users to upload the second data file
cv2 = st.file_uploader("Upload CV 2", type=["pdf"])

if cv1 is not None and cv2 is not None:
    st.write("Files successfully loaded!")

    def generate_text_from_pdf(pdf_file):
        # Step 1: Read the PDF file
        pdfreader = PdfReader(pdf_file)
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        # Step 2: Split the text to ensure it doesn't exceed token size
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Step 3: Replace OpenAIEmbeddings with Ollama Embeddings (Ollama model)
        #embeddings = Ollama(model="qwen2.5:latest")  # <-- Use Ollama's Qwen 2.5 model for embedding
        #embeddings = OllamaEmbeddings(model='qwen2.5:latest')
        #embeddings = OpenAIEmbeddings() # <-- Use Ollama's Qwen 2.5 model for embedding
        embeddings = OllamaEmbeddings(model='qwen2.5:latest')


        # Step 4: Create the document search index using FAISS
        document_search = FAISS.from_texts(texts, embeddings)

        return document_search

    candidates = ['candidate_1', 'candidate_2']

    st.title("Search CVs for ")
    Hr_question_1 = st.text_input("Enter your question below :", value='Which companies have you worked in and for how long?')
    Hr_question_2 = st.text_input("Enter your question below :", value='What are the Python libraries you know? Enumerate 3.')
    Hr_question_3 = st.text_input("Enter your question below :", value='Email and contact number.')

    # Build database to store answers
    df = pd.DataFrame(columns=[f'{Hr_question_1}', f'{Hr_question_2}', f'{Hr_question_3}'], index=candidates)

    # Generate FAISS embedding document search for each document
    document_search_1 = generate_text_from_pdf(pdf_file=cv1)
    document_search_2 = generate_text_from_pdf(pdf_file=cv2)

    # Run queries for each candidate and update the dataframe as we process it
    document_search_list = [document_search_1, document_search_2]

    # Initialize Ollama's Qwen 2.5 model in the chain
    chain = load_qa_chain(Ollama(model="qwen2.5:latest"), chain_type="stuff")  # <-- Use Ollama's Qwen 2.5 model here

    for candidate, document_search in zip(candidates, document_search_list):
        for query in [Hr_question_1, Hr_question_2, Hr_question_3]:
            docs = document_search.similarity_search(query)
            df.loc[candidate, query] = chain.run(input_documents=docs, question=query)

    st.title('Final output')
    st.write(df)
