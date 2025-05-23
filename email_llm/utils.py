"""
Utility functions for Email RAG Assistant
"""

import re
import html
import email
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import streamlit as st
import pandas as pd
from email.header import decode_header

def clean_email_text(text: str) -> str:
    """Clean and normalize email text"""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove email signatures (common patterns)
    signature_patterns = [
        r'\n--\s*\n.*',  # Standard signature delimiter
        r'\nSent from my.*',  # Mobile signatures
        r'\nGet Outlook for.*',  # Outlook signatures
        r'\n\[.*\]$',  # Bracketed signatures
    ]
    
    for pattern in signature_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove quoted text (replies)
    quoted_patterns = [
        r'\nOn .* wrote:.*',  # Standard reply format
        r'\n>.*',  # Quoted lines
        r'\nFrom:.*\nSent:.*\nTo:.*\nSubject:.*',  # Outlook reply format
    ]
    
    for pattern in quoted_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()

def extract_email_addresses(text: str) -> List[str]:
    """Extract email addresses from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text"""
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b',  # 1234567890
    ]
    
    phone_numbers = []
    for pattern in phone_patterns:
        phone_numbers.extend(re.findall(pattern, text))
    
    return phone_numbers

def extract_dates(text: str) -> List[str]:
    """Extract dates from text"""
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return dates

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def decode_email_header(header: str) -> str:
    """Decode email header with proper encoding handling"""
    if not header:
        return ""
    
    try:
        decoded_parts = decode_header(header)
        decoded_string = ""
        
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    decoded_string += part.decode(encoding, errors='ignore')
                else:
                    decoded_string += part.decode('utf-8', errors='ignore')
            else:
                decoded_string += part
        
        return decoded_string.strip()
    except Exception:
        return str(header)

def format_email_date(date_str: str) -> Optional[datetime]:
    """Parse and format email date string"""
    if not date_str:
        return None
    
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except Exception:
        # Try alternative parsing methods
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S',
            '%d %b %Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None

def calculate_email_stats(emails: List[Dict]) -> Dict[str, any]:
    """Calculate comprehensive email statistics"""
    if not emails:
        return {}
    
    # Basic stats
    total_emails = len(emails)
    unique_senders = len(set(email.get('sender', '') for email in emails))
    
    # Date range
    dates = [email.get('date') for email in emails if email.get('date')]
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range_days = (max_date - min_date).days
    else:
        min_date = max_date = None
        date_range_days = 0
    
    # Content stats
    total_chars = sum(len(email.get('body', '')) for email in emails)
    avg_length = total_chars / total_emails if total_emails > 0 else 0
    
    # Sender analysis
    sender_counts = {}
    for email in emails:
        sender = email.get('sender', 'Unknown')
        sender_counts[sender] = sender_counts.get(sender, 0) + 1
    
    top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Daily distribution
    daily_counts = {}
    for email in emails:
        if email.get('date'):
            date_key = email['date'].date()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
    
    return {
        'total_emails': total_emails,
        'unique_senders': unique_senders,
        'date_range_days': date_range_days,
        'min_date': min_date,
        'max_date': max_date,
        'avg_length': avg_length,
        'total_chars': total_chars,
        'top_senders': top_senders,
        'daily_counts': daily_counts,
        'sender_counts': sender_counts
    }

def create_email_dataframe(emails: List[Dict]) -> pd.DataFrame:
    """Convert emails to pandas DataFrame for analysis"""
    if not emails:
        return pd.DataFrame()
    
    df_data = []
    for email in emails:
        df_data.append({
            'subject': email.get('subject', ''),
            'sender': email.get('sender', ''),
            'date': email.get('date'),
            'body_length': len(email.get('body', '')),
            'has_attachments': 'attachment' in email.get('body', '').lower(),
            'is_reply': email.get('subject', '').lower().startswith(('re:', 'fwd:')),
            'message_id': email.get('id', '')
        })
    
    df = pd.DataFrame(df_data)
    
    # Add derived columns
    if 'date' in df.columns and not df['date'].isna().all():
        df['date_only'] = df['date'].dt.date
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
    
    return df

def validate_email_address(email_address: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email_address) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def highlight_search_terms(text: str, search_terms: List[str]) -> str:
    """Highlight search terms in text for display"""
    if not search_terms:
        return text
    
    highlighted_text = text
    for term in search_terms:
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark style="background-color: yellow;">{term}</mark>',
                highlighted_text
            )
    
    return highlighted_text

@st.cache_data
def get_sample_questions() -> List[str]:
    """Get sample questions for the chat interface"""
    return [
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
        "What are people asking me to do in my emails?"
    ]

def format_chat_message(message: str, max_length: int = 1000) -> str:
    """Format chat message for display"""
    # Clean up the message
    message = message.strip()
    
    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "..."
    
    # Add line breaks for better readability
    message = re.sub(r'(\. )([A-Z])', r'\1\n\n\2', message)
    
    return message