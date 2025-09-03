from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.helper import load_files, text_split, filter_minimal_docs, download_embedding
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_files("data/")
filtered_docs = filter_minimal_docs(extracted_data)
splitted_text_chunks = text_split(filtered_docs)
embedding = download_embedding()

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384,metric="cosine",spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(documents=splitted_text_chunks, embedding=embedding, index_name=index_name)
# docsearch = PineconeVectorStore.from_existing_index( embedding=embedding, index_name=index_name)