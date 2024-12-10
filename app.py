from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from main import CustomerSupportChatbot  # Import your chatbot class
import asyncio
from dotenv import load_dotenv
import os


app = FastAPI()

# Mount static files for CSS, JavaScript, and assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize chatbot globally
chatbot = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the chatbot model asynchronously when the FastAPI server starts.
    """
    global chatbot
    # Load environment variables from the .env file (if present)
    load_dotenv()
    api_key = os.getenv('API_KEY')
    overwrite_embeddings = False  # Change to True to regenerate embeddings

    # Initialize the chatbot instance
    chatbot = CustomerSupportChatbot(api_key=api_key, overwrite_embeddings=overwrite_embeddings)

    # Start loading embeddings in the background
    asyncio.create_task(load_embeddings())

async def load_embeddings():
    """
    Load embeddings asynchronously in the background, without blocking the main thread.
    """
    try:
        # Generate embeddings in a separate thread to avoid blocking the FastAPI app
        await asyncio.to_thread(chatbot.generate_embeddings, load_documents=True)
        print("Embeddings loaded successfully!")
    except Exception as e:
        print(f"Error loading embeddings: {e}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the main HTML page.
    """
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_bot(query: Query):
    """
    Handle chatbot queries from the client.
    """
    if chatbot is None:
        return JSONResponse(content={"error": "Chatbot is not initialized."}, status_code=500)

    try:
        # Make sure query processing is non-blocking
        response = await asyncio.to_thread(chatbot.query_chatbot, query.message)
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
