# 📧 Email RAG Assistant

A powerful Streamlit application that connects to your email account, processes your inbox using RAG (Retrieval-Augmented Generation), and allows you to chat with your emails using AI.

## 🚀 Features

### 📧 Email Integration
- **Multi-Provider Support**: Gmail, Outlook, Yahoo, and custom IMAP servers
- **Secure Connection**: Uses IMAP with SSL encryption
- **Flexible Fetching**: Configure number of emails and date range
- **Smart Processing**: Handles both plain text and HTML emails

### 🧠 RAG System
- **Vector Storage**: Uses FAISS for efficient email embeddings
- **Intelligent Chunking**: Splits emails into optimal chunks for retrieval
- **Context-Aware**: Maintains conversation history for better responses
- **Source Attribution**: Shows which emails were used to generate answers

### 💬 AI Chat Interface
- **Natural Language Queries**: Ask questions about your emails in plain English
- **Conversation Memory**: Maintains context across multiple questions
- **Source Documents**: View the actual emails used to generate responses
- **Sample Questions**: Pre-built queries to get you started

### 📊 Email Analytics
- **Statistics Dashboard**: Total emails, unique senders, date ranges
- **Visual Charts**: Daily email trends and top senders
- **Interactive Plots**: Powered by Plotly for rich visualizations

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI_World/email_llm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

4. **Run the application**:
   ```bash
   streamlit run email_rag_app.py
   ```

## ⚙️ Configuration

### 🔑 API Keys
- **OpenAI API Key**: Required for embeddings and chat functionality
- Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)

### 📧 Email Setup

#### Gmail
1. Enable 2-factor authentication
2. Generate an app-specific password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use the app password in the application

#### Outlook/Hotmail
1. Enable IMAP in account settings
2. Use your regular password or create an app password

#### Yahoo
1. Enable IMAP in account settings
2. Generate an app password for third-party applications

### 🔧 IMAP Settings

| Provider | IMAP Server | Port |
|----------|-------------|------|
| Gmail | imap.gmail.com | 993 |
| Outlook | outlook.office365.com | 993 |
| Yahoo | imap.mail.yahoo.com | 993 |

## 🎯 Usage

### 1. Initial Setup
1. Enter your OpenAI API key in the sidebar
2. Configure your email credentials
3. Select your email provider or enter custom IMAP settings
4. Set fetch parameters (number of emails, date range)

### 2. Connect to Email
1. Click "Connect to Email" button
2. Wait for emails to be fetched and processed
3. RAG system will be automatically created

### 3. Chat with Your Emails
Ask questions like:
- "What are the most important emails from this week?"
- "Who sent me the most emails?"
- "Are there any urgent emails I should respond to?"
- "What meetings are mentioned in my emails?"
- "Show me emails about project updates"
- "Summarize my recent conversations with John"

### 4. View Analytics
- Check email statistics and trends
- Analyze sender patterns
- View daily email volume

## 🏗️ Architecture

### Core Components

1. **EmailConnector**: Handles IMAP connection and email fetching
2. **EmailRAG**: Manages vector storage and LLM interactions
3. **Streamlit UI**: Provides interactive web interface

### Data Flow

```
Email Account → IMAP → Email Processing → Text Extraction → 
Chunking → Embeddings → Vector Store → RAG Chain → LLM → Response
```

### Technical Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: OpenAI Embeddings
- **LLM**: OpenAI GPT-3.5-turbo
- **Email Processing**: imaplib, email, html2text
- **Visualization**: Plotly

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: Emails are processed locally, not stored permanently
- **Secure Connection**: Uses SSL/TLS for email connections
- **API Security**: OpenAI API calls are made securely
- **No Persistence**: Email data is not saved to disk

### Best Practices
- Use app-specific passwords instead of main account passwords
- Keep your OpenAI API key secure
- Regularly rotate credentials
- Monitor API usage

## 📊 Features in Detail

### Email Processing
- **HTML to Text**: Converts HTML emails to readable text
- **Encoding Handling**: Properly decodes various character encodings
- **Attachment Filtering**: Focuses on email content, ignores attachments
- **Metadata Extraction**: Captures sender, subject, date, and message ID

### RAG Implementation
- **Chunking Strategy**: Uses RecursiveCharacterTextSplitter for optimal chunks
- **Embedding Model**: OpenAI text-embedding-ada-002
- **Retrieval**: Top-k similarity search with configurable parameters
- **Memory**: Conversation buffer for context retention

### Chat Interface
- **Streaming Responses**: Real-time response generation
- **Source Attribution**: Links answers to specific emails
- **Conversation History**: Maintains chat context
- **Error Handling**: Graceful error messages and recovery

## 🚀 Advanced Usage

### Custom Queries
```python
# Example queries you can ask:
"Find emails from my manager about the quarterly review"
"What deadlines are mentioned in my recent emails?"
"Summarize all emails about the new product launch"
"Who needs a response from me this week?"
```

### Filtering Options
- Date range filtering
- Sender-based filtering
- Subject keyword filtering
- Email folder selection

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check email credentials
   - Verify IMAP settings
   - Ensure app passwords are used for Gmail

2. **No Emails Fetched**
   - Check date range settings
   - Verify folder name (INBOX vs Inbox)
   - Check email account permissions

3. **RAG System Errors**
   - Verify OpenAI API key
   - Check internet connection
   - Ensure sufficient API credits

4. **Performance Issues**
   - Reduce number of emails fetched
   - Decrease chunk size
   - Use smaller embedding models

### Error Messages
- **"Failed to connect to email"**: Check credentials and IMAP settings
- **"No emails found"**: Adjust date range or folder settings
- **"Error creating vector store"**: Check OpenAI API key and connectivity

## 📈 Performance Optimization

### For Large Email Volumes
- Implement pagination for email fetching
- Use batch processing for embeddings
- Consider using local embedding models
- Implement caching for frequently accessed data

### Memory Management
- Process emails in batches
- Clear conversation memory periodically
- Use efficient data structures
- Monitor memory usage

## 🔮 Future Enhancements

### Planned Features
- **Multi-folder Support**: Process multiple email folders
- **Advanced Filtering**: Complex query filters
- **Email Categorization**: Automatic email classification
- **Response Generation**: Draft email responses
- **Calendar Integration**: Extract and manage calendar events
- **Attachment Processing**: Handle PDF and document attachments

### Technical Improvements
- **Local LLM Support**: Integration with local models
- **Database Storage**: Persistent email storage option
- **API Endpoints**: REST API for programmatic access
- **Mobile Interface**: Responsive design improvements

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review common error messages
3. Open an issue in the repository
4. Provide detailed error logs and configuration

---

**⚠️ Important**: This application requires valid email credentials and OpenAI API access. Ensure you have the necessary permissions and API credits before using.