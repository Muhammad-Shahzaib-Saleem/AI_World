"""
Configuration settings for Email RAG Assistant
"""

import os
from typing import Dict, Any

# Default IMAP server configurations
IMAP_SERVERS = {
    "Gmail": {
        "server": "imap.gmail.com",
        "port": 993,
        "ssl": True,
        "instructions": "Use app-specific password. Enable 2FA first."
    },
    "Outlook": {
        "server": "outlook.office365.com", 
        "port": 993,
        "ssl": True,
        "instructions": "Enable IMAP in account settings."
    },
    "Yahoo": {
        "server": "imap.mail.yahoo.com",
        "port": 993, 
        "ssl": True,
        "instructions": "Generate app password for third-party apps."
    },
    "iCloud": {
        "server": "imap.mail.me.com",
        "port": 993,
        "ssl": True,
        "instructions": "Use app-specific password."
    }
}

# LangChain configuration
LANGCHAIN_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-ada-002",
    "llm_model": "gpt-3.5-turbo",
    "temperature": 0.1,
    "max_tokens": 1000,
    "retrieval_k": 5
}

# Email processing configuration
EMAIL_CONFIG = {
    "default_limit": 100,
    "max_limit": 1000,
    "default_days_back": 30,
    "max_days_back": 365,
    "default_folder": "INBOX",
    "supported_folders": ["INBOX", "Sent", "Drafts", "Spam", "Trash"]
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "📧 Email RAG Assistant",
    "page_icon": "📧",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Security settings
SECURITY_CONFIG = {
    "max_email_size": 10 * 1024 * 1024,  # 10MB per email
    "timeout": 30,  # seconds
    "max_retries": 3,
    "rate_limit": 100  # requests per minute
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "imap_servers": IMAP_SERVERS,
        "langchain": LANGCHAIN_CONFIG,
        "email": EMAIL_CONFIG,
        "streamlit": STREAMLIT_CONFIG,
        "security": SECURITY_CONFIG
    }