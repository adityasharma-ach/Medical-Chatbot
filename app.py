from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_promt
from src.helper import download_embedding
import os
from dotenv import load_dotenv

load_dotenv()

# API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Setup FastAPI app
app = FastAPI()

# Mount static folder (for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja templates
templates = Jinja2Templates(directory="templates")

# Embeddings + Pinecone
embedding = download_embedding()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(embedding=embedding, index_name=index_name)

# LLM + RAG Chain
chatmodel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_promt),
    ("human", "{input}")
])
retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k":3})
question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

# Root route -> serve frontend
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Request model
class ChatRequest(BaseModel):
    message: str

# Chat API
@app.post("/api/chat/")
async def chat(request: ChatRequest):
    try:
        response = rag_chain.invoke({"input": request.message})
        return {"answer": response["answer"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


