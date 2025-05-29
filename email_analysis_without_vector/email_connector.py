import imaplib
import email
from email.header import decode_header
import ssl
from typing import List, Dict, Optional
import re
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailConnector:
    def __init__(self, email_address: str, password: str, imap_server: str = "imap.gmail.com", port: int = 993):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.port = port
        self.mail = None
        
    def connect(self) -> bool:
        """Connect to the email server"""
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect to server
            self.mail = imaplib.IMAP4_SSL(self.imap_server, self.port, ssl_context=context)
            
            # Login
            self.mail.login(self.email_address, self.password)
            logger.info(f"Successfully connected to {self.email_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the email server"""
        if self.mail:
            try:
                self.mail.close()
                self.mail.logout()
                logger.info("Disconnected from email server")
            except:
                pass
    
    def get_folders(self) -> List[str]:
        """Get list of available folders"""
        try:
            status, folders = self.mail.list()
            folder_list = []
            for folder in folders:
                folder_name = folder.decode().split('"')[3] if '"' in folder.decode() else folder.decode().split()[-1]
                folder_list.append(folder_name)
            return folder_list
        except Exception as e:
            logger.error(f"Error getting folders: {str(e)}")
            return []
    
    def select_folder(self, folder: str = "INBOX") -> bool:
        """Select a folder to work with"""
        try:
            status, messages = self.mail.select(folder)
            if status == 'OK':
                logger.info(f"Selected folder: {folder}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error selecting folder {folder}: {str(e)}")
            return False
    
    def search_emails(self, criteria: str = "ALL", limit: int = 50) -> List[str]:
        """Search for emails based on criteria"""
        try:
            status, messages = self.mail.search(None, criteria)
            if status == 'OK':
                email_ids = messages[0].split()
                # Return latest emails first, limited by the limit parameter
                return email_ids[-limit:] if len(email_ids) > limit else email_ids
            return []
        except Exception as e:
            logger.error(f"Error searching emails: {str(e)}")
            return []
    
    def fetch_email(self, email_id: str) -> Optional[Dict]:
        """Fetch a single email by ID"""
        try:
            status, msg_data = self.mail.fetch(email_id, '(RFC822)')
            if status == 'OK':
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)
                
                # Extract email details
                subject = self._decode_header(email_message["Subject"])
                sender = self._decode_header(email_message["From"])
                recipient = self._decode_header(email_message["To"])
                date = email_message["Date"]
                
                # Extract body
                body = self._extract_body(email_message)
                
                return {
                    "id": email_id.decode() if isinstance(email_id, bytes) else email_id,
                    "subject": subject,
                    "sender": sender,
                    "recipient": recipient,
                    "date": date,
                    "body": body,
                    "raw_message": email_message
                }
        except Exception as e:
            logger.error(f"Error fetching email {email_id}: {str(e)}")
            return None
    
    def fetch_emails(self, folder: str = "INBOX", limit: int = 50, criteria: str = "ALL") -> List[Dict]:
        """Fetch multiple emails from a folder"""
        emails = []
        
        if not self.select_folder(folder):
            return emails
        
        email_ids = self.search_emails(criteria, limit)
        
        for email_id in reversed(email_ids):  # Most recent first
            email_data = self.fetch_email(email_id)
            if email_data:
                emails.append(email_data)
        
        return emails
    
    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        if header is None:
            return ""
        
        decoded_parts = decode_header(header)
        decoded_header = ""
        
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                try:
                    decoded_header += part.decode(encoding or 'utf-8')
                except:
                    decoded_header += part.decode('utf-8', errors='ignore')
            else:
                decoded_header += part
        
        return decoded_header
    
    def _extract_body(self, email_message) -> str:
        """Extract email body content"""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        continue
                elif content_type == "text/html" and "attachment" not in content_disposition and not body:
                    try:
                        html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        # Simple HTML to text conversion
                        body = re.sub('<[^<]+?>', '', html_body)
                    except:
                        continue
        else:
            try:
                body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(email_message.get_payload())
        
        return body.strip()
    
    def get_recent_emails(self, days: int = 7, folder: str = "INBOX") -> List[Dict]:
        """Get emails from the last N days"""
        date_criteria = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
        criteria = f'SINCE "{date_criteria}"'
        return self.fetch_emails(folder=folder, criteria=criteria)