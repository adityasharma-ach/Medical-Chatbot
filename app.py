from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from  langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_promt


app = FastAPI()
chatmodel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
promt = ChatPromptTemplate.from_messages([("sytem", system_promt),("human",{input})])
