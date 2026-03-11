from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os 

def load_documents():
    file_path = "data/documents/test.pdf"
    load_file = PyPDFLoader(file_path)
    document = load_file.load()
    return document

def chunk_text(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(document)
    return text_chunks

docs = chunk_text(load_documents())
print(len(docs))