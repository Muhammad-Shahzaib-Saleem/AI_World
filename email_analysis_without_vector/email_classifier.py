import openai
from typing import List, Dict, Optional
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailClassifier:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
        # Predefined categories
        self.categories = {
            "inbox": "Regular inbox emails",
            "spam": "Spam or unwanted promotional emails",
            "junk": "Junk emails or low-priority messages",
            "sent": "Emails sent by the user",
            "important": "Important or urgent emails",
            "personal": "Personal emails from friends/family",
            "work": "Work-related emails",
            "promotional": "Marketing and promotional emails",
            "social": "Social media notifications",
            "financial": "Banking, payment, and financial emails"
        }
    
    def classify_email(self, email_data: Dict) -> Dict:
        """Classify a single email into categories"""
        try:
            prompt = self._create_classification_prompt(email_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an email classification expert. Classify emails accurately based on their content, sender, and subject."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            classification_result = response.choices[0].message.content.strip()
            
            # Parse the result
            try:
                result = json.loads(classification_result)
            except json.JSONDecodeError:
                # Fallback parsing
                result = self._parse_classification_fallback(classification_result)
            
            # Add metadata
            result["classified_at"] = datetime.now().isoformat()
            result["email_id"] = email_data.get("id")
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying email: {str(e)}")
            return {
                "primary_category": "inbox",
                "confidence": 0.5,
                "secondary_categories": [],
                "reasoning": "Classification failed",
                "classified_at": datetime.now().isoformat(),
                "email_id": email_data.get("id")
            }
    
    def classify_emails_batch(self, emails: List[Dict]) -> List[Dict]:
        """Classify multiple emails"""
        classified_emails = []
        
        for email_data in emails:
            classification = self.classify_email(email_data)
            email_with_classification = email_data.copy()
            email_with_classification["classification"] = classification
            classified_emails.append(email_with_classification)
        
        return classified_emails
    
    def answer_query(self, query: str, emails: List[Dict]) -> str:
        """Answer natural language queries about emails"""
        try:
            # Prepare email summaries for context
            email_summaries = []
            for email in emails[:20]:  # Limit to prevent token overflow
                summary = {
                    "subject": email.get("subject", "")[:100],
                    "sender": email.get("sender", "")[:50],
                    "date": email.get("date", ""),
                    "category": email.get("classification", {}).get("primary_category", "unknown"),
                    "body_preview": email.get("body", "")[:200]
                }
                email_summaries.append(summary)
            
            prompt = f"""
            Based on the following email data, answer the user's query: "{query}"
            
            Email Data:
            {json.dumps(email_summaries, indent=2)}
            
            Provide a helpful and accurate response based on the email information available.
            If the query asks for specific emails, provide details about relevant emails.
            If the query asks for counts or statistics, provide accurate numbers.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an email assistant. Answer questions about emails accurately and helpfully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your query: {str(e)}"
    
    def filter_emails_by_query(self, query: str, emails: List[Dict]) -> List[Dict]:
        """Filter emails based on natural language query"""
        try:
            # Create a prompt to understand the filtering criteria
            prompt = f"""
            Based on this query: "{query}"
            
            Determine what filtering criteria should be applied to emails.
            Return a JSON object with filtering parameters:
            {{
                "categories": ["list of relevant categories"],
                "sender_keywords": ["keywords to match in sender"],
                "subject_keywords": ["keywords to match in subject"],
                "date_range": "recent/today/this_week/this_month/all",
                "content_keywords": ["keywords to match in email body"]
            }}
            
            Available categories: {list(self.categories.keys())}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an email filtering expert. Extract filtering criteria from natural language queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            filter_criteria = json.loads(response.choices[0].message.content.strip())
            
            # Apply filters
            filtered_emails = self._apply_filters(emails, filter_criteria)
            
            return filtered_emails
            
        except Exception as e:
            logger.error(f"Error filtering emails: {str(e)}")
            return emails  # Return all emails if filtering fails
    
    def _create_classification_prompt(self, email_data: Dict) -> str:
        """Create a prompt for email classification"""
        subject = email_data.get("subject", "")
        sender = email_data.get("sender", "")
        body = email_data.get("body", "")[:500]  # Limit body length
        
        prompt = f"""
        Classify this email into the most appropriate category:
        
        Subject: {subject}
        Sender: {sender}
        Body Preview: {body}
        
        Available Categories:
        {json.dumps(self.categories, indent=2)}
        
        Return a JSON response with:
        {{
            "primary_category": "most_appropriate_category",
            "confidence": 0.0-1.0,
            "secondary_categories": ["other_relevant_categories"],
            "reasoning": "brief explanation of classification"
        }}
        """
        
        return prompt
    
    def _parse_classification_fallback(self, text: str) -> Dict:
        """Fallback parsing if JSON parsing fails"""
        # Simple fallback - look for category keywords
        text_lower = text.lower()
        
        for category in self.categories.keys():
            if category in text_lower:
                return {
                    "primary_category": category,
                    "confidence": 0.7,
                    "secondary_categories": [],
                    "reasoning": "Fallback classification"
                }
        
        return {
            "primary_category": "inbox",
            "confidence": 0.5,
            "secondary_categories": [],
            "reasoning": "Default classification"
        }
    
    def _apply_filters(self, emails: List[Dict], criteria: Dict) -> List[Dict]:
        """Apply filtering criteria to emails"""
        filtered = emails
        
        # Filter by categories
        if criteria.get("categories"):
            filtered = [
                email for email in filtered
                if email.get("classification", {}).get("primary_category") in criteria["categories"]
            ]
        
        # Filter by sender keywords
        if criteria.get("sender_keywords"):
            sender_keywords = [kw.lower() for kw in criteria["sender_keywords"]]
            filtered = [
                email for email in filtered
                if any(kw in email.get("sender", "").lower() for kw in sender_keywords)
            ]
        
        # Filter by subject keywords
        if criteria.get("subject_keywords"):
            subject_keywords = [kw.lower() for kw in criteria["subject_keywords"]]
            filtered = [
                email for email in filtered
                if any(kw in email.get("subject", "").lower() for kw in subject_keywords)
            ]
        
        # Filter by content keywords
        if criteria.get("content_keywords"):
            content_keywords = [kw.lower() for kw in criteria["content_keywords"]]
            filtered = [
                email for email in filtered
                if any(kw in email.get("body", "").lower() for kw in content_keywords)
            ]
        
        return filtered