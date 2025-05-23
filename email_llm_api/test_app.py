"""
Simple test to verify FastAPI app structure
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import uvicorn

# Simple test models
class EmailConfig(BaseModel):
    email_address: EmailStr
    password: str
    imap_server: str = "imap.gmail.com"
    port: int = 993

class ChatMessage(BaseModel):
    message: str
    session_id: str

# Create FastAPI app
app = FastAPI(
    title="📧 Email RAG API",
    description="A REST API for email-based Retrieval-Augmented Generation using LangChain",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "📧 Email RAG API",
        "status": "running",
        "endpoints": [
            "/docs - API Documentation",
            "/health - Health Check",
            "/connect - Connect to Email",
            "/chat - Chat with Emails"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Email RAG API is running"
    }

@app.post("/connect")
async def connect_email(config: EmailConfig):
    return {
        "success": True,
        "message": f"Would connect to {config.email_address}",
        "session_id": "test-session-123"
    }

@app.post("/chat")
async def chat_with_emails(chat_message: ChatMessage):
    return {
        "answer": f"This is a test response to: {chat_message.message}",
        "sources": [],
        "session_id": chat_message.session_id
    }

if __name__ == "__main__":
    print("🚀 Starting Email RAG API test server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)