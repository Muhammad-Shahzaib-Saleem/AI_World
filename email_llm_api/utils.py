"""
Utility functions for Email RAG API
"""

import re
import html
import email
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from email.header import decode_header
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

def generate_hash(text: str) -> str:
    """Generate SHA-256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()

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
        r'\n_{3,}.*',  # Underline signatures
        r'\nBest regards.*',  # Common closings
        r'\nThanks.*\n.*@.*',  # Thanks with email
    ]
    
    for pattern in signature_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove quoted text (replies)
    quoted_patterns = [
        r'\nOn .* wrote:.*',  # Standard reply format
        r'\n>.*',  # Quoted lines
        r'\nFrom:.*\nSent:.*\nTo:.*\nSubject:.*',  # Outlook reply format
        r'\n_{5,}.*From:.*',  # Forwarded messages
    ]
    
    for pattern in quoted_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return text.strip()

def extract_email_addresses(text: str) -> List[str]:
    """Extract email addresses from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return list(set(re.findall(email_pattern, text)))

def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text"""
    phone_patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
        r'\b\d{10}\b',  # 1234567890
        r'\+\d{1,3}\s*\d{3,4}\s*\d{3,4}\s*\d{4}',  # International
    ]
    
    phone_numbers = []
    for pattern in phone_patterns:
        phone_numbers.extend(re.findall(pattern, text))
    
    return list(set(phone_numbers))

def extract_dates(text: str) -> List[str]:
    """Extract dates from text"""
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b',  # Day, Month DD
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return list(set(dates))

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text"""
    url_patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    ]
    
    urls = []
    for pattern in url_patterns:
        urls.extend(re.findall(pattern, text))
    
    return list(set(urls))

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text"""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter keywords
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    # Count frequency and return top keywords
    keyword_freq = {}
    for keyword in keywords:
        keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    # Sort by frequency and return top 20
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    return [keyword for keyword, freq in sorted_keywords[:20]]

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
    except Exception as e:
        logger.warning(f"Error decoding header: {e}")
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
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None

def calculate_email_stats(emails: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive email statistics"""
    if not emails:
        return {
            'total_emails': 0,
            'unique_senders': 0,
            'date_range_days': 0,
            'min_date': None,
            'max_date': None,
            'avg_length': 0,
            'total_chars': 0,
            'top_senders': [],
            'daily_counts': {},
            'sender_counts': {},
            'keywords': [],
            'email_addresses': [],
            'phone_numbers': [],
            'urls': []
        }
    
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
            date_key = email['date'].date().isoformat()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
    
    # Extract entities from all emails
    all_text = ' '.join(email.get('body', '') for email in emails)
    keywords = extract_keywords(all_text)
    email_addresses = extract_email_addresses(all_text)
    phone_numbers = extract_phone_numbers(all_text)
    urls = extract_urls(all_text)
    
    return {
        'total_emails': total_emails,
        'unique_senders': unique_senders,
        'date_range_days': date_range_days,
        'min_date': min_date.isoformat() if min_date else None,
        'max_date': max_date.isoformat() if max_date else None,
        'avg_length': avg_length,
        'total_chars': total_chars,
        'top_senders': [{'sender': sender, 'count': count} for sender, count in top_senders],
        'daily_counts': daily_counts,
        'sender_counts': sender_counts,
        'keywords': keywords,
        'email_addresses': email_addresses[:10],  # Limit to top 10
        'phone_numbers': phone_numbers[:10],
        'urls': urls[:10]
    }

def validate_email_address(email_address: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email_address) is not None

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format (UUID)"""
    pattern = r'^[a-fA-F0-9-]{36}$'
    return re.match(pattern, session_id) is not None

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

def extract_urgency_indicators(text: str) -> List[str]:
    """Extract urgency indicators from email text"""
    urgency_patterns = [
        r'\burgent\b',
        r'\basap\b',
        r'\bimmediate\b',
        r'\bpriority\b',
        r'\bdeadline\b',
        r'\btime.sensitive\b',
        r'\baction.required\b',
        r'\bresponse.needed\b',
        r'\bfyi\b',
        r'\bfor.your.information\b'
    ]
    
    indicators = []
    for pattern in urgency_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        indicators.extend(matches)
    
    return list(set(indicators))

def calculate_email_priority(email_data: Dict) -> int:
    """Calculate email priority score (1-10, 10 being highest)"""
    score = 5  # Base score
    
    subject = email_data.get('subject', '').lower()
    body = email_data.get('body', '').lower()
    sender = email_data.get('sender', '').lower()
    
    # Check for urgency indicators
    urgency_keywords = ['urgent', 'asap', 'immediate', 'priority', 'deadline']
    for keyword in urgency_keywords:
        if keyword in subject:
            score += 2
        elif keyword in body:
            score += 1
    
    # Check for action words
    action_keywords = ['action required', 'response needed', 'please respond', 'need your']
    for keyword in action_keywords:
        if keyword in subject:
            score += 1
        elif keyword in body:
            score += 0.5
    
    # Check sender importance (basic heuristic)
    important_domains = ['manager', 'director', 'ceo', 'president', 'admin']
    for domain in important_domains:
        if domain in sender:
            score += 1
            break
    
    # Check for meeting/calendar related
    meeting_keywords = ['meeting', 'calendar', 'schedule', 'appointment']
    for keyword in meeting_keywords:
        if keyword in subject or keyword in body:
            score += 0.5
            break
    
    # Ensure score is within bounds
    return min(10, max(1, int(score)))

def extract_action_items(text: str) -> List[str]:
    """Extract potential action items from email text"""
    action_patterns = [
        r'please\s+\w+.*?[.!?]',
        r'can\s+you\s+.*?[.!?]',
        r'could\s+you\s+.*?[.!?]',
        r'would\s+you\s+.*?[.!?]',
        r'need\s+you\s+to\s+.*?[.!?]',
        r'action\s+required:.*?[.!?]',
        r'todo:.*?[.!?]',
        r'task:.*?[.!?]'
    ]
    
    action_items = []
    for pattern in action_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        action_items.extend([match.strip() for match in matches])
    
    return action_items[:5]  # Limit to top 5

def async_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for async functions with retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

def rate_limit(calls_per_minute: int = 60):
    """Simple rate limiting decorator"""
    call_times = []
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = datetime.now()
            
            # Remove calls older than 1 minute
            call_times[:] = [call_time for call_time in call_times if now - call_time < timedelta(minutes=1)]
            
            # Check if we've exceeded the rate limit
            if len(call_times) >= calls_per_minute:
                raise Exception(f"Rate limit exceeded: {calls_per_minute} calls per minute")
            
            # Record this call
            call_times.append(now)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def create_email_summary(emails: List[Dict], max_emails: int = 5) -> str:
    """Create a summary of the most important emails"""
    if not emails:
        return "No emails to summarize."
    
    # Sort emails by priority
    prioritized_emails = []
    for email_data in emails:
        priority = calculate_email_priority(email_data)
        prioritized_emails.append((priority, email_data))
    
    prioritized_emails.sort(key=lambda x: x[0], reverse=True)
    
    summary_parts = []
    for i, (priority, email_data) in enumerate(prioritized_emails[:max_emails]):
        subject = email_data.get('subject', 'No Subject')
        sender = email_data.get('sender', 'Unknown Sender')
        date = email_data.get('date', datetime.now())
        
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d %H:%M')
        else:
            date_str = str(date)
        
        summary_parts.append(
            f"{i+1}. **{subject}** (Priority: {priority}/10)\n"
            f"   From: {sender}\n"
            f"   Date: {date_str}\n"
        )
    
    return "\n".join(summary_parts)

class EmailProcessor:
    """Advanced email processing utilities"""
    
    @staticmethod
    def extract_meeting_info(text: str) -> Dict[str, Any]:
        """Extract meeting information from email text"""
        meeting_info = {
            'has_meeting': False,
            'dates': [],
            'times': [],
            'locations': [],
            'attendees': []
        }
        
        # Check for meeting indicators
        meeting_keywords = ['meeting', 'call', 'conference', 'appointment', 'session']
        if any(keyword in text.lower() for keyword in meeting_keywords):
            meeting_info['has_meeting'] = True
        
        # Extract dates and times
        meeting_info['dates'] = extract_dates(text)
        
        # Extract times
        time_pattern = r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b'
        meeting_info['times'] = re.findall(time_pattern, text)
        
        # Extract locations (basic patterns)
        location_patterns = [
            r'room\s+\w+',
            r'conference\s+room\s+\w+',
            r'building\s+\w+',
            r'floor\s+\d+',
            r'zoom\s+link',
            r'teams\s+meeting'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            meeting_info['locations'].extend(matches)
        
        # Extract attendees (email addresses)
        meeting_info['attendees'] = extract_email_addresses(text)
        
        return meeting_info
    
    @staticmethod
    def categorize_email(email_data: Dict) -> str:
        """Categorize email based on content"""
        subject = email_data.get('subject', '').lower()
        body = email_data.get('body', '').lower()
        sender = email_data.get('sender', '').lower()
        
        # Define category keywords
        categories = {
            'meeting': ['meeting', 'call', 'conference', 'appointment', 'calendar'],
            'project': ['project', 'task', 'milestone', 'deliverable', 'sprint'],
            'urgent': ['urgent', 'asap', 'immediate', 'priority', 'deadline'],
            'financial': ['budget', 'cost', 'expense', 'invoice', 'payment', 'financial'],
            'hr': ['hr', 'human resources', 'benefits', 'vacation', 'leave', 'policy'],
            'technical': ['bug', 'issue', 'error', 'code', 'development', 'technical'],
            'marketing': ['marketing', 'campaign', 'promotion', 'advertisement', 'social media'],
            'sales': ['sales', 'customer', 'client', 'deal', 'proposal', 'quote'],
            'administrative': ['admin', 'policy', 'procedure', 'announcement', 'notice'],
            'personal': ['personal', 'private', 'confidential', 'family', 'health']
        }
        
        # Check each category
        for category, keywords in categories.items():
            if any(keyword in subject or keyword in body for keyword in keywords):
                return category
        
        # Check sender-based categories
        if any(domain in sender for domain in ['noreply', 'no-reply', 'automated']):
            return 'automated'
        
        return 'general'