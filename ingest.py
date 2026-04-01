from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
import os 

def load_documents():
    file_path = "data/documents/test.pdf"
    load_file = PyPDFLoader(file_path)
    document = load_file.load()
    return document

# Chunks text into smaller text for LLMs to handle
def split_documents(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(document)
    return text_chunks

# Text chunks are converted into embeddings (vectors) and stored in a vector db (Chroma).
def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory = "vector_db")
    vector_store.persist()



docs = load_documents()
chunks = split_documents(docs)
create_vector_store(chunks)
#print(len(chunks))