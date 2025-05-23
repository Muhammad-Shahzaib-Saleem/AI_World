"""
FastAPI Email RAG Application
A REST API for email-based Retrieval-Augmented Generation using LangChain
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import logging
from contextlib import asynccontextmanager

# Pydantic models
from pydantic import BaseModel, EmailStr, Field, validator

# Email and RAG imports
import imaplib
import email
from email.header import decode_header
import html2text
from bs4 import BeautifulSoup

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for storing RAG systems per session
rag_sessions: Dict[str, Any] = {}

# Pydantic Models
class EmailConfig(BaseModel):
    email_address: EmailStr
    password: str
    imap_server: str = "imap.gmail.com"
    port: int = 993
    folder: str = "INBOX"
    limit: int = Field(default=100, ge=1, le=1000)
    days_back: int = Field(default=30, ge=1, le=365)

class ChatMessage(BaseModel):
    message: str
    session_id: str

class EmailStats(BaseModel):
    total_emails: int
    unique_senders: int
    date_range_days: int
    avg_length: float
    top_senders: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str

class ConnectionResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    stats: Optional[EmailStats] = None

# Email Connector Class
class EmailConnector:
    """Handle email connection and retrieval"""
    
    def __init__(self):
        self.connection = None
        self.email_data = []
    
    async def connect_to_email(self, config: EmailConfig) -> bool:
        """Connect to email server using IMAP"""
        try:
            self.connection = imaplib.IMAP4_SSL(config.imap_server, config.port)
            self.connection.login(config.email_address, config.password)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to email: {str(e)}")
            return False
    
    async def fetch_emails(self, config: EmailConfig) -> List[Dict]:
        """Fetch emails from specified folder"""
        if not self.connection:
            return []
        
        try:
            self.connection.select(config.folder)
            
            # Calculate date range
            since_date = (datetime.now() - timedelta(days=config.days_back)).strftime("%d-%b-%Y")
            
            # Search for emails
            status, messages = self.connection.search(None, f'SINCE {since_date}')
            email_ids = messages[0].split()
            
            # Limit the number of emails
            email_ids = email_ids[-config.limit:] if len(email_ids) > config.limit else email_ids
            
            emails = []
            for email_id in email_ids:
                try:
                    status, msg_data = self.connection.fetch(email_id, "(RFC822)")
                    email_message = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email details
                    email_info = self._extract_email_info(email_message)
                    if email_info:
                        emails.append(email_info)
                        
                except Exception as e:
                    logger.warning(f"Error processing email {email_id}: {str(e)}")
                    continue
            
            self.email_data = emails
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching emails: {str(e)}")
            return []
    
    def _extract_email_info(self, email_message) -> Dict:
        """Extract information from email message"""
        try:
            # Get basic email info
            subject = self._decode_header(email_message.get("Subject", ""))
            sender = self._decode_header(email_message.get("From", ""))
            date = email_message.get("Date", "")
            message_id = email_message.get("Message-ID", "")
            
            # Extract body
            body = self._extract_body(email_message)
            
            # Parse date
            try:
                parsed_date = email.utils.parsedate_to_datetime(date)
            except:
                parsed_date = datetime.now()
            
            return {
                "id": message_id,
                "subject": subject,
                "sender": sender,
                "date": parsed_date,
                "body": body,
                "raw_date": date
            }
            
        except Exception as e:
            logger.warning(f"Error extracting email info: {str(e)}")
            return None
    
    def _decode_header(self, header) -> str:
        """Decode email header"""
        if header:
            decoded = decode_header(header)
            return ''.join([
                part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part
                for part, encoding in decoded
            ])
        return ""
    
    def _extract_body(self, email_message) -> str:
        """Extract email body content"""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if "attachment" not in content_disposition:
                    if content_type == "text/plain":
                        charset = part.get_content_charset() or 'utf-8'
                        body += part.get_payload(decode=True).decode(charset, errors='ignore')
                    elif content_type == "text/html":
                        charset = part.get_content_charset() or 'utf-8'
                        html_content = part.get_payload(decode=True).decode(charset, errors='ignore')
                        # Convert HTML to text
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        body += h.handle(html_content)
        else:
            content_type = email_message.get_content_type()
            charset = email_message.get_content_charset() or 'utf-8'
            
            if content_type == "text/plain":
                body = email_message.get_payload(decode=True).decode(charset, errors='ignore')
            elif content_type == "text/html":
                html_content = email_message.get_payload(decode=True).decode(charset, errors='ignore')
                h = html2text.HTML2Text()
                h.ignore_links = True
                body = h.handle(html_content)
        
        return body.strip()
    
    def disconnect(self):
        """Close email connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
            except:
                pass

# Email RAG Class
class EmailRAG:
    """RAG system for email analysis"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    async def create_vectorstore(self, emails: List[Dict]) -> bool:
        """Create vector store from emails"""
        try:
            # Convert emails to documents
            documents = []
            for email_data in emails:
                # Create comprehensive document content
                content = f"""
Subject: {email_data['subject']}
From: {email_data['sender']}
Date: {email_data['date'].strftime('%Y-%m-%d %H:%M:%S')}

{email_data['body']}
                """.strip()
                
                # Create metadata
                metadata = {
                    "subject": email_data['subject'],
                    "sender": email_data['sender'],
                    "date": email_data['date'].isoformat(),
                    "message_id": email_data['id']
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            # Create QA chain
            await self._create_qa_chain()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False
    
    async def _create_qa_chain(self):
        """Create conversational QA chain"""
        # Custom prompt template
        prompt_template = """
You are an intelligent email assistant. Use the following email context to answer questions about the user's emails.
Be specific and cite relevant emails when possible. If you can't find relevant information, say so clearly.

Context from emails:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a helpful and accurate answer based on the email context:
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
    
    async def query(self, question: str) -> Dict[str, Any]:
        """Query the email RAG system"""
        if not self.qa_chain:
            return {"error": "RAG system not initialized"}
        
        try:
            result = self.qa_chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "chat_history": result.get("chat_history", [])
            }
            
        except Exception as e:
            return {"error": f"Error querying RAG system: {str(e)}"}

# Utility functions
def calculate_email_stats(emails: List[Dict]) -> EmailStats:
    """Calculate email statistics"""
    if not emails:
        return EmailStats(
            total_emails=0,
            unique_senders=0,
            date_range_days=0,
            avg_length=0,
            top_senders=[]
        )
    
    total_emails = len(emails)
    unique_senders = len(set(email.get('sender', '') for email in emails))
    
    # Date range
    dates = [email.get('date') for email in emails if email.get('date')]
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range_days = (max_date - min_date).days
    else:
        date_range_days = 0
    
    # Content stats
    total_chars = sum(len(email.get('body', '')) for email in emails)
    avg_length = total_chars / total_emails if total_emails > 0 else 0
    
    # Sender analysis
    sender_counts = {}
    for email in emails:
        sender = email.get('sender', 'Unknown')
        sender_counts[sender] = sender_counts.get(sender, 0) + 1
    
    top_senders = [
        {"sender": sender, "count": count}
        for sender, count in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    return EmailStats(
        total_emails=total_emails,
        unique_senders=unique_senders,
        date_range_days=date_range_days,
        avg_length=avg_length,
        top_senders=top_senders
    )

def generate_session_id() -> str:
    """Generate unique session ID"""
    import uuid
    return str(uuid.uuid4())

# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Email RAG API...")
    yield
    # Shutdown
    logger.info("Shutting down Email RAG API...")
    # Clean up sessions
    global rag_sessions
    for session_id in list(rag_sessions.keys()):
        if "email_connector" in rag_sessions[session_id]:
            rag_sessions[session_id]["email_connector"].disconnect()
    rag_sessions.clear()

app = FastAPI(
    title="📧 Email RAG API",
    description="A REST API for email-based Retrieval-Augmented Generation using LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_openai_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify OpenAI API key"""
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI API key required"
        )
    return credentials.credentials

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <html>
        <head>
            <title>📧 Email RAG API</title>
        </head>
        <body>
            <h1>📧 Email RAG API</h1>
            <p>A REST API for email-based Retrieval-Augmented Generation using LangChain</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">📚 API Documentation (Swagger)</a></li>
                <li><a href="/redoc">📖 API Documentation (ReDoc)</a></li>
                <li><a href="/frontend">🖥️ Frontend Interface</a></li>
            </ul>
            <h2>Quick Start:</h2>
            <ol>
                <li>Connect to email: POST /connect</li>
                <li>Chat with emails: POST /chat</li>
                <li>Get statistics: GET /stats/{session_id}</li>
            </ol>
        </body>
    </html>
    """

@app.post("/connect", response_model=ConnectionResponse)
async def connect_email(
    config: EmailConfig,
    background_tasks: BackgroundTasks,
    openai_key: str = Depends(verify_openai_key)
):
    """Connect to email account and initialize RAG system"""
    session_id = generate_session_id()
    
    try:
        # Create email connector
        email_connector = EmailConnector()
        
        # Connect to email
        if not await email_connector.connect_to_email(config):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to connect to email account"
            )
        
        # Fetch emails
        emails = await email_connector.fetch_emails(config)
        
        if not emails:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No emails found in the specified time range"
            )
        
        # Create RAG system
        email_rag = EmailRAG(openai_key)
        if not await email_rag.create_vectorstore(emails):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create RAG system"
            )
        
        # Store session
        rag_sessions[session_id] = {
            "email_connector": email_connector,
            "email_rag": email_rag,
            "emails": emails,
            "config": config,
            "created_at": datetime.now()
        }
        
        # Calculate stats
        stats = calculate_email_stats(emails)
        
        return ConnectionResponse(
            success=True,
            message=f"Successfully connected and processed {len(emails)} emails",
            session_id=session_id,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in connect_email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_emails(
    chat_message: ChatMessage,
    openai_key: str = Depends(verify_openai_key)
):
    """Chat with emails using RAG system"""
    session_id = chat_message.session_id
    
    if session_id not in rag_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found. Please connect to email first."
        )
    
    try:
        email_rag = rag_sessions[session_id]["email_rag"]
        result = await email_rag.query(chat_message.message)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Format source documents
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "subject": doc.metadata.get("subject", "N/A"),
                "sender": doc.metadata.get("sender", "N/A"),
                "date": doc.metadata.get("date", "N/A"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_emails: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )

@app.get("/stats/{session_id}", response_model=EmailStats)
async def get_email_stats(session_id: str):
    """Get email statistics for a session"""
    if session_id not in rag_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    emails = rag_sessions[session_id]["emails"]
    return calculate_email_stats(emails)

@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    sessions = []
    for session_id, session_data in rag_sessions.items():
        sessions.append({
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "email_count": len(session_data["emails"]),
            "email_address": session_data["config"].email_address
        })
    
    return {"sessions": sessions}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in rag_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Disconnect email
    if "email_connector" in rag_sessions[session_id]:
        rag_sessions[session_id]["email_connector"].disconnect()
    
    # Remove session
    del rag_sessions[session_id]
    
    return {"message": "Session deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(rag_sessions)
    }

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions for the chat interface"""
    return {
        "questions": [
            "What are the most important emails from this week?",
            "Who sent me the most emails?",
            "Are there any urgent emails I should respond to?",
            "What meetings or events are mentioned in my emails?",
            "Show me emails about project updates",
            "What are the main topics discussed in my recent emails?",
            "Find emails that need my immediate attention",
            "What deadlines are mentioned in my emails?",
            "Show me emails about budget or financial matters",
            "Find emails containing attachments"
        ]
    }

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )