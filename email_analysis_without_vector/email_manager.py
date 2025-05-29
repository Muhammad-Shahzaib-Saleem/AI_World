from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import logging
from email_connector import EmailConnector
from email_classifier import EmailClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailManager:
    def __init__(self, email_address: str, email_password: str, openai_api_key: str):
        self.email_connector = EmailConnector(email_address, email_password)
        self.email_classifier = EmailClassifier(openai_api_key)
        self.cached_emails = {}
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to email server"""
        self.is_connected = self.email_connector.connect()
        return self.is_connected
    
    def disconnect(self):
        """Disconnect from email server"""
        self.email_connector.disconnect()
        self.is_connected = False
    
    def fetch_and_classify_emails(self, folder: str = "INBOX", limit: int = 50, force_refresh: bool = False) -> List[Dict]:
        """Fetch emails from specified folder and classify them"""
        if not self.is_connected:
            logger.error("Not connected to email server")
            return []
        
        cache_key = f"{folder}_{limit}"
        
        # Check cache first
        if not force_refresh and cache_key in self.cached_emails:
            logger.info(f"Returning cached emails for {folder}")
            return self.cached_emails[cache_key]
        
        logger.info(f"Fetching emails from {folder}...")
        
        # Fetch emails
        emails = self.email_connector.fetch_emails(folder=folder, limit=limit)
        
        if not emails:
            logger.warning(f"No emails found in {folder}")
            return []
        
        logger.info(f"Classifying {len(emails)} emails...")
        
        # Classify emails
        classified_emails = self.email_classifier.classify_emails_batch(emails)
        
        # Cache the results
        self.cached_emails[cache_key] = classified_emails
        
        logger.info(f"Successfully processed {len(classified_emails)} emails from {folder}")
        return classified_emails
    
    def get_emails_by_category(self, category: str, folder: str = "INBOX", limit: int = 50) -> List[Dict]:
        """Get emails filtered by category"""
        emails = self.fetch_and_classify_emails(folder=folder, limit=limit)
        
        filtered_emails = [
            email for email in emails
            if email.get("classification", {}).get("primary_category") == category
        ]
        
        return filtered_emails
    
    def search_emails_by_query(self, query: str, folder: str = "INBOX", limit: int = 50) -> List[Dict]:
        """Search emails using natural language query"""
        emails = self.fetch_and_classify_emails(folder=folder, limit=limit)
        return self.email_classifier.filter_emails_by_query(query, emails)
    
    def answer_email_query(self, query: str, folder: str = "INBOX", limit: int = 50) -> str:
        """Answer natural language questions about emails"""
        emails = self.fetch_and_classify_emails(folder=folder, limit=limit)
        return self.email_classifier.answer_query(query, emails)
    
    def get_email_statistics(self, folder: str = "INBOX", limit: int = 50) -> Dict:
        """Get statistics about emails"""
        emails = self.fetch_and_classify_emails(folder=folder, limit=limit)
        
        if not emails:
            return {"total": 0, "categories": {}, "senders": {}}
        
        # Count by categories
        category_counts = {}
        sender_counts = {}
        
        for email in emails:
            # Category statistics
            category = email.get("classification", {}).get("primary_category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Sender statistics
            sender = email.get("sender", "unknown")
            # Extract email address from sender
            if "<" in sender and ">" in sender:
                sender = sender.split("<")[1].split(">")[0]
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        # Get top senders
        top_senders = dict(sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            "total": len(emails),
            "categories": category_counts,
            "top_senders": top_senders,
            "folder": folder
        }
    
    def get_available_folders(self) -> List[str]:
        """Get list of available email folders"""
        if not self.is_connected:
            return []
        return self.email_connector.get_folders()
    
    def refresh_cache(self, folder: str = "INBOX", limit: int = 50):
        """Refresh cached emails for a folder"""
        cache_key = f"{folder}_{limit}"
        if cache_key in self.cached_emails:
            del self.cached_emails[cache_key]
        return self.fetch_and_classify_emails(folder=folder, limit=limit, force_refresh=True)
    
    def export_emails(self, emails: List[Dict], filename: str = None) -> str:
        """Export emails to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emails_export_{timestamp}.json"
        
        try:
            # Prepare data for export (remove raw_message for cleaner export)
            export_data = []
            for email in emails:
                email_copy = email.copy()
                if "raw_message" in email_copy:
                    del email_copy["raw_message"]
                export_data.append(email_copy)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Exported {len(emails)} emails to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting emails: {str(e)}")
            return None
    
    def get_recent_emails(self, days: int = 7, folder: str = "INBOX") -> List[Dict]:
        """Get emails from the last N days"""
        if not self.is_connected:
            return []
        
        emails = self.email_connector.get_recent_emails(days=days, folder=folder)
        if emails:
            return self.email_classifier.classify_emails_batch(emails)
        return []