# 📧 Email RAG API

A FastAPI-based REST API for email-based Retrieval-Augmented Generation (RAG) with a Streamlit frontend interface.

## 🚀 Features

### 🔧 FastAPI Backend (`app.py`)
- **RESTful API**: Complete REST API for email RAG operations
- **Session Management**: Multi-user session handling with unique session IDs
- **Email Integration**: IMAP support for Gmail, Outlook, Yahoo, and custom servers
- **RAG System**: LangChain-powered retrieval-augmented generation
- **Vector Storage**: FAISS for efficient email embeddings
- **Authentication**: Bearer token authentication with OpenAI API keys
- **Background Tasks**: Async processing for better performance
- **Health Monitoring**: Health check and session management endpoints

### 🖥️ Streamlit Frontend (`frontend.py`)
- **Web Interface**: User-friendly web interface for the API
- **Real-time Chat**: Interactive chat interface with email context
- **Analytics Dashboard**: Email statistics and visualizations
- **Session Management**: Connect, manage, and delete sessions
- **API Integration**: Seamless communication with FastAPI backend

## 📁 Project Structure

```
email_llm_api/
├── app.py              # FastAPI backend application
├── frontend.py         # Streamlit frontend interface
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This documentation
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
cd email_llm_api
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)

### 3. Start the Backend

```bash
# Development mode with auto-reload
python app.py

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Frontend

```bash
# In a new terminal
streamlit run frontend.py --server.port 8501
```

## 🔗 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and documentation links |
| `POST` | `/connect` | Connect to email account and initialize RAG |
| `POST` | `/chat` | Send chat messages to RAG system |
| `GET` | `/stats/{session_id}` | Get email statistics for session |
| `GET` | `/sessions` | List all active sessions |
| `DELETE` | `/sessions/{session_id}` | Delete a specific session |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/sample-questions` | Get sample questions for chat |

### Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📊 API Usage Examples

### 1. Connect to Email

```bash
curl -X POST "http://localhost:8000/connect" \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "email_address": "your.email@gmail.com",
    "password": "your_app_password",
    "imap_server": "imap.gmail.com",
    "port": 993,
    "limit": 100,
    "days_back": 30
  }'
```

Response:
```json
{
  "success": true,
  "message": "Successfully connected and processed 100 emails",
  "session_id": "uuid-session-id",
  "stats": {
    "total_emails": 100,
    "unique_senders": 25,
    "date_range_days": 30,
    "avg_length": 1250.5,
    "top_senders": [...]
  }
}
```

### 2. Chat with Emails

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the most important emails from this week?",
    "session_id": "your-session-id"
  }'
```

Response:
```json
{
  "answer": "Based on your emails from this week, here are the most important ones...",
  "sources": [
    {
      "subject": "Project Update - Urgent",
      "sender": "manager@company.com",
      "date": "2024-01-15T10:30:00",
      "content_preview": "The project deadline has been moved up..."
    }
  ],
  "session_id": "your-session-id"
}
```

### 3. Get Statistics

```bash
curl -X GET "http://localhost:8000/stats/your-session-id" \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY"
```

## 🔧 Configuration

### Email Providers

| Provider | IMAP Server | Port | Notes |
|----------|-------------|------|-------|
| Gmail | imap.gmail.com | 993 | Requires app-specific password |
| Outlook | outlook.office365.com | 993 | Enable IMAP in settings |
| Yahoo | imap.mail.yahoo.com | 993 | Generate app password |

### API Configuration

```python
# app.py configuration
LANGCHAIN_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-ada-002",
    "llm_model": "gpt-3.5-turbo",
    "temperature": 0.1,
    "retrieval_k": 5
}
```

## 🖥️ Frontend Usage

### 1. Access the Frontend

Navigate to: `http://localhost:8501`

### 2. Configure Settings

1. **API Settings**: Set the FastAPI backend URL
2. **OpenAI API Key**: Enter your OpenAI API key
3. **Email Credentials**: Configure your email account
4. **Fetch Settings**: Set email limit and date range

### 3. Connect and Chat

1. Click "Connect to Email" to initialize the session
2. View email statistics and analytics
3. Use the chat interface to ask questions about your emails
4. Try sample questions or ask custom queries

## 🔒 Security Features

### Authentication
- **Bearer Token**: Uses OpenAI API key as authentication
- **Session Isolation**: Each session is isolated with unique IDs
- **Secure Headers**: CORS and security headers configured

### Data Privacy
- **Local Processing**: Emails processed in memory, not stored permanently
- **Session Cleanup**: Automatic cleanup on shutdown
- **Secure Connections**: SSL/TLS for email connections

### Best Practices
- Use environment variables for sensitive data
- Implement rate limiting for production use
- Use HTTPS in production
- Regularly rotate API keys

## 🚀 Deployment

### Development
```bash
# Backend
python app.py

# Frontend
streamlit run frontend.py
```

### Production

#### Using Docker
```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Backend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Using Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
  
  frontend:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      - api
```

## 📈 Performance Optimization

### Backend Optimization
- **Async Processing**: Uses async/await for I/O operations
- **Connection Pooling**: Efficient IMAP connection management
- **Caching**: Session-based caching for email data
- **Background Tasks**: Non-blocking operations

### Frontend Optimization
- **Session State**: Efficient state management
- **Lazy Loading**: Load data only when needed
- **Caching**: Streamlit caching for API responses

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check if FastAPI backend is running
   - Verify API URL in frontend configuration
   - Check firewall and network settings

2. **Email Connection Failed**
   - Verify email credentials
   - Check IMAP server settings
   - Ensure app passwords are used for Gmail

3. **OpenAI API Errors**
   - Verify API key is valid
   - Check API usage limits
   - Ensure sufficient credits

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### Session Monitoring
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/sessions
```

## 🔮 Future Enhancements

### Planned Features
- **Database Integration**: Persistent storage for sessions
- **User Authentication**: Multi-user support with proper auth
- **Email Categorization**: Automatic email classification
- **Attachment Processing**: Handle PDF and document attachments
- **Real-time Updates**: WebSocket support for real-time chat
- **Advanced Analytics**: More detailed email insights

### Technical Improvements
- **Caching Layer**: Redis for session and data caching
- **Load Balancing**: Support for multiple API instances
- **Monitoring**: Prometheus metrics and logging
- **Testing**: Comprehensive test suite

## 📝 API Schema

### Request Models

```python
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
```

### Response Models

```python
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
```

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check logs for error details
4. Open an issue in the repository

---

**⚠️ Important**: This application requires valid email credentials and OpenAI API access. Ensure you have the necessary permissions and API credits before using.