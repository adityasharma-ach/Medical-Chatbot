#  Libraries import for chatbot
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document 
from langchain_pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings

# Function to load files
def load_files(data):
    loader = DirectoryLoader(data, glob="*pdf",loader_cls=PyPDFLoader)
    document = loader.load()
    return document

extracted_data = load_files("data/")

# function reduces metadata of documents so that only the "source" field is kept
def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source":src}))
    return minimal_docs

filtered_docs = filter_minimal_docs(extracted_data)

# function for Split text into chunks
def text_split(filtered_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunk = text_splitter.split_documents(filtered_data)
    return text_chunk

splitted_text_chunks = text_split(filtered_docs)

# Function for setup embedding model
def download_embedding():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embedding = download_embedding()


