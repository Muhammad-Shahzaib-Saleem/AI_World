# Email Analysis Without Vector DB

A powerful email analysis application that uses LLM-based classification and natural language processing to analyze emails without requiring a vector database. This approach provides faster processing for real-time email analysis.

## 🚀 Features

- **Real-time Email Fetching**: Connect to your email account and fetch emails instantly
- **AI-Powered Classification**: Automatically classify emails into categories (inbox, spam, junk, important, etc.)
- **Natural Language Search**: Search emails using natural language queries
- **Email Analytics**: View statistics and visualizations of your email data
- **Chat Interface**: Ask questions about your emails in natural language
- **Multiple Interfaces**: Both web UI (Streamlit) and CLI versions available
- **Export Functionality**: Export emails to JSON or CSV formats
- **No Vector DB Required**: Direct LLM processing for faster analysis

## 📋 Prerequisites

- Python 3.8+
- Email account with IMAP access (Gmail recommended)
- OpenAI API key
- For Gmail: App Password (requires 2-factor authentication)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   cd AI_World/email_analysis_without_vector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your credentials:
   ```
   EMAIL_ADDRESS=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   OPENAI_API_KEY=your_openai_api_key
   ```

## 🔧 Gmail Setup

1. **Enable 2-Factor Authentication** on your Google account
2. **Generate App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate a new app password for "Mail"
   - Use this password in the `EMAIL_PASSWORD` field

## 🚀 Usage

### Web Interface (Streamlit)

```bash
streamlit run app.py --server.port 12000 --server.address 0.0.0.0
```

Access the application at: `http://localhost:12000`

### CLI Interface

```bash
python cli_demo.py
```

## 📱 Web Interface Features

### 1. Email Browser
- Browse emails from different folders (INBOX, Sent, Spam, etc.)
- View email classifications with confidence scores
- Filter emails by category
- Detailed email view with full content

### 2. Search & Query
- **Natural Language Search**: "show me important emails from last week"
- **Category Filtering**: Filter by specific categories
- **Advanced Queries**: Complex search criteria

### 3. Analytics Dashboard
- Email distribution by category (pie chart)
- Top senders analysis (bar chart)
- Detailed statistics and metrics
- Category breakdown with percentages

### 4. Chat Interface
- Ask questions about your emails
- Example queries:
  - "How many emails did I receive today?"
  - "Who sent me the most emails?"
  - "What are the main topics in my recent emails?"
  - "Show me all emails from my boss"

### 5. Settings & Export
- Export emails to JSON or CSV
- Cache management
- Application settings

## 🎯 Email Categories

The system automatically classifies emails into these categories:

- **inbox**: Regular inbox emails
- **spam**: Spam or unwanted promotional emails
- **junk**: Junk emails or low-priority messages
- **sent**: Emails sent by the user
- **important**: Important or urgent emails
- **personal**: Personal emails from friends/family
- **work**: Work-related emails
- **promotional**: Marketing and promotional emails
- **social**: Social media notifications
- **financial**: Banking, payment, and financial emails

## 🔍 Natural Language Query Examples

### Search Queries
- "Find all emails from John"
- "Show me promotional emails from this week"
- "Get important emails I haven't read"
- "Find emails about meetings"

### Question Queries
- "How many spam emails did I receive today?"
- "What's the most common type of email I get?"
- "Who are my top 5 email contacts?"
- "Are there any urgent emails I should check?"

## 🏗️ Architecture

```
email_analysis_without_vector/
├── app.py                 # Streamlit web application
├── cli_demo.py           # Command-line interface
├── email_connector.py    # Email fetching and IMAP handling
├── email_classifier.py   # LLM-based email classification
├── email_manager.py      # Main email management logic
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

### Key Components

1. **EmailConnector**: Handles IMAP connection and email fetching
2. **EmailClassifier**: Uses OpenAI GPT for email classification and queries
3. **EmailManager**: Orchestrates email operations and caching
4. **Streamlit App**: Web interface with multiple tabs and features
5. **CLI Demo**: Command-line interface for testing

## ⚡ Performance Features

- **Smart Caching**: Emails are cached to avoid re-processing
- **Batch Processing**: Efficient batch classification of emails
- **No Vector DB**: Direct LLM processing eliminates vector database overhead
- **Configurable Limits**: Control number of emails processed
- **Real-time Processing**: Fresh email analysis without pre-indexing

## 🔒 Security

- Environment variables for sensitive credentials
- No email storage (processed in memory)
- Secure IMAP SSL connections
- API key protection

## 🐛 Troubleshooting

### Common Issues

1. **Gmail Connection Failed**:
   - Ensure 2-factor authentication is enabled
   - Use App Password, not regular password
   - Check if IMAP is enabled in Gmail settings

2. **OpenAI API Errors**:
   - Verify API key is correct
   - Check API usage limits
   - Ensure sufficient credits

3. **Slow Performance**:
   - Reduce email limit for faster processing
   - Use cache when possible
   - Check internet connection

### Error Messages

- `"Failed to connect"`: Check email credentials
- `"Classification failed"`: Check OpenAI API key
- `"No emails found"`: Check folder name and email account

## 📊 Example Outputs

### Email Classification
```json
{
  "primary_category": "work",
  "confidence": 0.95,
  "secondary_categories": ["important"],
  "reasoning": "Email from manager about project deadline"
}
```

### Statistics
```
Total emails: 150
Categories:
  • work: 45 (30.0%)
  • personal: 30 (20.0%)
  • promotional: 25 (16.7%)
  • spam: 20 (13.3%)
  • important: 15 (10.0%)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Test with CLI version first

## 🔮 Future Enhancements

- Support for more email providers
- Advanced filtering options
- Email threading analysis
- Sentiment analysis
- Automated email responses
- Integration with calendar and tasks
- Mobile-responsive design
- Multi-language support