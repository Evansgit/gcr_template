import os
import signal
from google.cloud import storage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_community import GCSDirectoryLoader
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, validator
from fastapi import FastAPI, Request, Body, HTTPException
from dotenv import load_dotenv

load_dotenv()

# Explicitly set credentials from the service account JSON file
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"

# Get API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# If API key is not set in environment variables, use the default value from constants
if not api_key:
    api_key = "AIzaSyCmT0QZxBc83s9HOR952UNBlcJTIaIJdN4"  # Replace with your actual API key

llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=api_key, convert_system_message_to_human=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

output_parser = StrOutputParser()

# Use the service account credentials explicitly
client = storage.Client()

loader = GCSDirectoryLoader(project_name="graphite-ally-412021", bucket="langchain_bucket_euclide")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context. If the context does not have relevant information, please provide an answer based on your knowledge:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# FastAPI App Setup
app = FastAPI()

# Define Input/Output Models (using Pydantic for validation)
class ChatInput(BaseModel):
    user_input: str
    chat_history: list[str]  
    user_context: str 

class ChatResponse(BaseModel):
    answer: str
    updated_chat_history: list[str]

# Endpoint to handle chat requests
@app.post("/chat")
async def chat_endpoint(request: Request, chat_input: ChatInput = Body(...)):
    try:
        # Call your LangChain processing logic 
        response, updated_chat_history = process_user_input(
            chat_input.user_input, 
            chat_input.chat_history, 
            chat_input.user_context
        )
        # Update chat history
        chat_histories = updated_chat_history
        return ChatResponse(answer=response, updated_chat_history=updated_chat_history)
    except Exception as e:
        # Handle the exception gracefully
        error_message = f"An error occurred: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

def process_user_input(user_input, chat_history, user_context):
    chat_history_langchain = [
        HumanMessage(content=chat_history[i]) if i % 2 == 0 else AIMessage(content=chat_history[i])
        for i in range(len(chat_history))
    ]
    try:
        response = retrieval_chain.invoke({
            "context": user_context,
            "chat_history": chat_history_langchain,
            "input": user_input
        })
    except Exception as e:
        # Handle the exception gracefully
        error_message = f"Error occurred during LangChain processing: {str(e)}"
        raise Exception(error_message)
    chat_history_langchain.append(HumanMessage(content=user_input))
    chat_history_langchain.append(AIMessage(content=response['answer']))
    updated_chat_history_strings = [message.content for message in chat_history_langchain]
    return response['answer'], updated_chat_history_strings
