"""
Configuration settings for Email RAG API
"""

import os
from typing import Dict, Any

# FastAPI Configuration
FASTAPI_CONFIG = {
    "title": "📧 Email RAG API",
    "description": "A REST API for email-based Retrieval-Augmented Generation using LangChain",
    "version": "1.0.0",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("DEBUG", "True").lower() == "true",
    "reload": os.getenv("RELOAD", "True").lower() == "true"
}

# CORS Configuration
CORS_CONFIG = {
    "allow_origins": ["*"],  # Configure appropriately for production
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

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
    },
    "Custom": {
        "server": "custom",
        "port": 993,
        "ssl": True,
        "instructions": "Enter your custom IMAP server details."
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
    "retrieval_k": 5,
    "memory_key": "chat_history",
    "return_messages": True,
    "output_key": "answer"
}

# Email processing configuration
EMAIL_CONFIG = {
    "default_limit": 100,
    "max_limit": 1000,
    "min_limit": 1,
    "default_days_back": 30,
    "max_days_back": 365,
    "min_days_back": 1,
    "default_folder": "INBOX",
    "supported_folders": ["INBOX", "Sent", "Drafts", "Spam", "Trash"],
    "timeout": 30,  # seconds
    "max_retries": 3
}

# Security settings
SECURITY_CONFIG = {
    "max_email_size": 10 * 1024 * 1024,  # 10MB per email
    "session_timeout": 3600,  # 1 hour in seconds
    "max_sessions_per_user": 5,
    "rate_limit": 100,  # requests per minute
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
    "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "email_rag_api.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "email_rag_api": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

# Sample questions for the chat interface
SAMPLE_QUESTIONS = [
    "What are the most important emails from this week?",
    "Who sent me the most emails?",
    "Are there any urgent emails I should respond to?",
    "What meetings or events are mentioned in my emails?",
    "Show me emails about project updates",
    "What are the main topics discussed in my recent emails?",
    "Find emails that need my immediate attention",
    "Summarize my conversations with [specific person]",
    "What deadlines are mentioned in my emails?",
    "Show me emails about budget or financial matters",
    "Find emails containing attachments",
    "What are people asking me to do in my emails?",
    "Show me emails from my manager",
    "Find emails about meetings this week",
    "What are the latest project updates?",
    "Show me emails that mention 'urgent' or 'important'",
    "Find emails about travel or vacation",
    "What emails discuss financial or budget topics?",
    "Show me emails with action items for me",
    "Find emails about upcoming events or conferences"
]

# API response templates
RESPONSE_TEMPLATES = {
    "connection_success": "Successfully connected and processed {count} emails",
    "connection_failed": "Failed to connect to email account: {error}",
    "no_emails_found": "No emails found in the specified time range",
    "rag_creation_failed": "Failed to create RAG system: {error}",
    "session_not_found": "Session not found. Please connect to email first.",
    "chat_error": "Error processing chat message: {error}",
    "session_deleted": "Session deleted successfully",
    "health_check": "Email RAG API is healthy"
}

# Validation rules
VALIDATION_RULES = {
    "email_address": {
        "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "max_length": 254
    },
    "password": {
        "min_length": 1,
        "max_length": 128
    },
    "imap_server": {
        "pattern": r'^[a-zA-Z0-9.-]+$',
        "max_length": 253
    },
    "session_id": {
        "pattern": r'^[a-fA-F0-9-]{36}$'  # UUID format
    },
    "chat_message": {
        "min_length": 1,
        "max_length": 2000
    }
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "fastapi": FASTAPI_CONFIG,
        "cors": CORS_CONFIG,
        "imap_servers": IMAP_SERVERS,
        "langchain": LANGCHAIN_CONFIG,
        "email": EMAIL_CONFIG,
        "security": SECURITY_CONFIG,
        "logging": LOGGING_CONFIG,
        "sample_questions": SAMPLE_QUESTIONS,
        "response_templates": RESPONSE_TEMPLATES,
        "validation_rules": VALIDATION_RULES
    }

def get_imap_server_config(provider: str) -> Dict[str, Any]:
    """Get IMAP server configuration for a specific provider"""
    return IMAP_SERVERS.get(provider, IMAP_SERVERS["Custom"])

def get_sample_questions() -> list:
    """Get sample questions for the chat interface"""
    return SAMPLE_QUESTIONS.copy()

def validate_email_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Validate email configuration and return errors if any"""
    errors = {}
    
    # Validate email address
    import re
    email_pattern = VALIDATION_RULES["email_address"]["pattern"]
    if not re.match(email_pattern, config.get("email_address", "")):
        errors["email_address"] = "Invalid email address format"
    
    # Validate password
    password = config.get("password", "")
    if len(password) < VALIDATION_RULES["password"]["min_length"]:
        errors["password"] = "Password is required"
    
    # Validate IMAP server
    imap_server = config.get("imap_server", "")
    if not imap_server or imap_server == "custom":
        errors["imap_server"] = "IMAP server is required"
    
    # Validate port
    port = config.get("port", 0)
    if not isinstance(port, int) or port < 1 or port > 65535:
        errors["port"] = "Port must be between 1 and 65535"
    
    # Validate limits
    limit = config.get("limit", 0)
    if limit < EMAIL_CONFIG["min_limit"] or limit > EMAIL_CONFIG["max_limit"]:
        errors["limit"] = f"Limit must be between {EMAIL_CONFIG['min_limit']} and {EMAIL_CONFIG['max_limit']}"
    
    days_back = config.get("days_back", 0)
    if days_back < EMAIL_CONFIG["min_days_back"] or days_back > EMAIL_CONFIG["max_days_back"]:
        errors["days_back"] = f"Days back must be between {EMAIL_CONFIG['min_days_back']} and {EMAIL_CONFIG['max_days_back']}"
    
    return errors