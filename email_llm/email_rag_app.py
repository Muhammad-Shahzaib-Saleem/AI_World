import streamlit as st
import imaplib
import email
from email.header import decode_header
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import re
import html2text
from bs4 import BeautifulSoup
import json

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="📧 Email RAG Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EmailConnector:
    """Handle email connection and retrieval"""
    
    def __init__(self):
        self.connection = None
        self.email_data = []
    
    def connect_to_email(self, email_address: str, password: str, imap_server: str, port: int = 993) -> bool:
        """Connect to email server using IMAP"""
        try:
            self.connection = imaplib.IMAP4_SSL(imap_server, port)
            self.connection.login(email_address, password)
            return True
        except Exception as e:
            st.error(f"Failed to connect to email: {str(e)}")
            return False
    
    def fetch_emails(self, folder: str = "INBOX", limit: int = 100, days_back: int = 30) -> List[Dict]:
        """Fetch emails from specified folder"""
        if not self.connection:
            return []
        
        try:
            self.connection.select(folder)
            
            # Calculate date range
            since_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")
            
            # Search for emails
            status, messages = self.connection.search(None, f'SINCE {since_date}')
            email_ids = messages[0].split()
            
            # Limit the number of emails
            email_ids = email_ids[-limit:] if len(email_ids) > limit else email_ids
            
            emails = []
            for email_id in email_ids:
                try:
                    status, msg_data = self.connection.fetch(email_id, "(RFC822)")
                    email_message = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email details
                    email_info = self._extract_email_info(email_message)
                    if email_info:
                        emails.append(email_info)
                        
                except Exception as e:
                    st.warning(f"Error processing email {email_id}: {str(e)}")
                    continue
            
            self.email_data = emails
            return emails
            
        except Exception as e:
            st.error(f"Error fetching emails: {str(e)}")
            return []
    
    def _extract_email_info(self, email_message) -> Dict:
        """Extract information from email message"""
        try:
            # Get basic email info
            subject = self._decode_header(email_message.get("Subject", ""))
            sender = self._decode_header(email_message.get("From", ""))
            date = email_message.get("Date", "")
            message_id = email_message.get("Message-ID", "")
            
            # Extract body
            body = self._extract_body(email_message)
            
            # Parse date
            try:
                parsed_date = email.utils.parsedate_to_datetime(date)
            except:
                parsed_date = datetime.now()
            
            return {
                "id": message_id,
                "subject": subject,
                "sender": sender,
                "date": parsed_date,
                "body": body,
                "raw_date": date
            }
            
        except Exception as e:
            st.warning(f"Error extracting email info: {str(e)}")
            return None
    
    def _decode_header(self, header) -> str:
        """Decode email header"""
        if header:
            decoded = decode_header(header)
            return ''.join([
                part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part
                for part, encoding in decoded
            ])
        return ""
    
    def _extract_body(self, email_message) -> str:
        """Extract email body content"""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if "attachment" not in content_disposition:
                    if content_type == "text/plain":
                        charset = part.get_content_charset() or 'utf-8'
                        body += part.get_payload(decode=True).decode(charset, errors='ignore')
                    elif content_type == "text/html":
                        charset = part.get_content_charset() or 'utf-8'
                        html_content = part.get_payload(decode=True).decode(charset, errors='ignore')
                        # Convert HTML to text
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        body += h.handle(html_content)
        else:
            content_type = email_message.get_content_type()
            charset = email_message.get_content_charset() or 'utf-8'
            
            if content_type == "text/plain":
                body = email_message.get_payload(decode=True).decode(charset, errors='ignore')
            elif content_type == "text/html":
                html_content = email_message.get_payload(decode=True).decode(charset, errors='ignore')
                h = html2text.HTML2Text()
                h.ignore_links = True
                body = h.handle(html_content)
        
        return body.strip()
    
    def disconnect(self):
        """Close email connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
            except:
                pass

class EmailRAG:
    """RAG system for email analysis"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def create_vectorstore(self, emails: List[Dict]) -> bool:
        """Create vector store from emails"""
        try:
            # Convert emails to documents
            documents = []
            for email_data in emails:
                # Create comprehensive document content
                content = f"""
Subject: {email_data['subject']}
From: {email_data['sender']}
Date: {email_data['date'].strftime('%Y-%m-%d %H:%M:%S')}

{email_data['body']}
                """.strip()
                
                # Create metadata
                metadata = {
                    "subject": email_data['subject'],
                    "sender": email_data['sender'],
                    "date": email_data['date'].isoformat(),
                    "message_id": email_data['id']
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            # Create QA chain
            self._create_qa_chain()
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def _create_qa_chain(self):
        """Create conversational QA chain"""
        # Custom prompt template
        prompt_template = """
You are an intelligent email assistant. Use the following email context to answer questions about the user's emails.
Be specific and cite relevant emails when possible. If you can't find relevant information, say so clearly.

Context from emails:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a helpful and accurate answer based on the email context:
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the email RAG system"""
        if not self.qa_chain:
            return {"error": "RAG system not initialized"}
        
        try:
            result = self.qa_chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "chat_history": result.get("chat_history", [])
            }
            
        except Exception as e:
            return {"error": f"Error querying RAG system: {str(e)}"}

def main():
    st.title("📧 Email RAG Assistant")
    st.markdown("**Connect to your email and chat with your inbox using AI**")
    st.markdown("---")
    
    # Initialize session state
    if "email_connector" not in st.session_state:
        st.session_state.email_connector = EmailConnector()
    
    if "email_rag" not in st.session_state:
        st.session_state.email_rag = None
    
    if "emails_loaded" not in st.session_state:
        st.session_state.emails_loaded = False
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # OpenAI API Key
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key"
    )
    
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    # Email connection settings
    st.sidebar.subheader("📧 Email Connection")
    
    email_address = st.sidebar.text_input(
        "Email Address",
        value=os.getenv("EMAIL_ADDRESS", ""),
        placeholder="your.email@gmail.com"
    )
    
    email_password = st.sidebar.text_input(
        "Email Password/App Password",
        type="password",
        value=os.getenv("EMAIL_PASSWORD", ""),
        help="Use app-specific password for Gmail"
    )
    
    # IMAP server settings
    imap_servers = {
        "Gmail": "imap.gmail.com",
        "Outlook/Hotmail": "outlook.office365.com",
        "Yahoo": "imap.mail.yahoo.com",
        "Custom": "custom"
    }
    
    selected_provider = st.sidebar.selectbox("Email Provider", list(imap_servers.keys()))
    
    if selected_provider == "Custom":
        imap_server = st.sidebar.text_input("IMAP Server", placeholder="imap.example.com")
    else:
        imap_server = imap_servers[selected_provider]
        st.sidebar.info(f"IMAP Server: {imap_server}")
    
    imap_port = st.sidebar.number_input("IMAP Port", value=993, min_value=1, max_value=65535)
    
    # Email fetch settings
    st.sidebar.subheader("📥 Fetch Settings")
    email_limit = st.sidebar.slider("Number of emails to fetch", 10, 500, 100)
    days_back = st.sidebar.slider("Days back to fetch", 1, 365, 30)
    
    # Connect to email
    if st.sidebar.button("🔗 Connect to Email", type="primary"):
        if not email_address or not email_password:
            st.error("Please enter both email address and password.")
            return
        
        with st.spinner("Connecting to email..."):
            if st.session_state.email_connector.connect_to_email(
                email_address, email_password, imap_server, imap_port
            ):
                st.success("✅ Connected to email successfully!")
                
                # Fetch emails
                with st.spinner("Fetching emails..."):
                    emails = st.session_state.email_connector.fetch_emails(
                        limit=email_limit, days_back=days_back
                    )
                
                if emails:
                    st.success(f"✅ Fetched {len(emails)} emails!")
                    
                    # Initialize RAG system
                    with st.spinner("Creating RAG system..."):
                        st.session_state.email_rag = EmailRAG(openai_api_key)
                        if st.session_state.email_rag.create_vectorstore(emails):
                            st.session_state.emails_loaded = True
                            st.success("✅ RAG system created successfully!")
                        else:
                            st.error("❌ Failed to create RAG system.")
                else:
                    st.warning("No emails found in the specified time range.")
    
    # Main content
    if st.session_state.emails_loaded:
        # Display email statistics
        emails = st.session_state.email_connector.email_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", len(emails))
        with col2:
            unique_senders = len(set(email['sender'] for email in emails))
            st.metric("Unique Senders", unique_senders)
        with col3:
            avg_length = sum(len(email['body']) for email in emails) / len(emails) if emails else 0
            st.metric("Avg Email Length", f"{avg_length:.0f} chars")
        with col4:
            date_range = max(email['date'] for email in emails) - min(email['date'] for email in emails)
            st.metric("Date Range", f"{date_range.days} days")
        
        # Email analysis charts
        st.subheader("📊 Email Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Emails by date
            df = pd.DataFrame(emails)
            df['date_only'] = df['date'].dt.date
            daily_counts = df.groupby('date_only').size().reset_index(name='count')
            
            fig = px.line(daily_counts, x='date_only', y='count', 
                         title="Emails per Day", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top senders
            sender_counts = df['sender'].value_counts().head(10)
            fig = px.bar(x=sender_counts.values, y=sender_counts.index, 
                        orientation='h', title="Top 10 Senders")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Chat interface
        st.subheader("💬 Chat with Your Emails")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📎 Source Emails"):
                        for i, source in enumerate(message["sources"]):
                            st.write(f"**Email {i+1}:**")
                            st.write(f"Subject: {source.metadata.get('subject', 'N/A')}")
                            st.write(f"From: {source.metadata.get('sender', 'N/A')}")
                            st.write(f"Date: {source.metadata.get('date', 'N/A')}")
                            st.write("---")
        
        # Chat input
        if prompt := st.chat_input("Ask about your emails..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from RAG system
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.email_rag.query(prompt)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(result["answer"])
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result.get("source_documents", [])
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Show source documents
                    if result.get("source_documents"):
                        with st.expander("📎 Source Emails"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.write(f"**Email {i+1}:**")
                                st.write(f"Subject: {doc.metadata.get('subject', 'N/A')}")
                                st.write(f"From: {doc.metadata.get('sender', 'N/A')}")
                                st.write(f"Date: {doc.metadata.get('date', 'N/A')}")
                                st.write("---")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the most important emails from this week?",
            "Who sent me the most emails?",
            "Are there any urgent emails I should respond to?",
            "What meetings or events are mentioned in my emails?",
            "Show me emails about project updates",
            "What are the main topics discussed in my recent emails?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = cols[i % 2]
            if col.button(question, key=f"sample_{i}"):
                # Trigger the question
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.rerun()
    
    else:
        # Instructions
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Enter your OpenAI API Key** in the sidebar
        2. **Configure your email settings** in the sidebar
        3. **Click 'Connect to Email'** to fetch your emails
        4. **Start chatting** with your email data!
        
        ### 📧 Email Provider Setup:
        
        **Gmail:**
        - Enable 2-factor authentication
        - Generate an app-specific password
        - Use the app password instead of your regular password
        
        **Outlook/Hotmail:**
        - Enable IMAP in your account settings
        - Use your regular password or app password
        
        **Yahoo:**
        - Enable IMAP in your account settings
        - Generate an app password for third-party apps
        """)
        
        st.info("💡 **Tip:** This app processes your emails locally and doesn't store them permanently. Your data remains private!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>📧 Email RAG Assistant | Built with Streamlit & LangChain</p>
        <p>Your emails are processed securely and not stored permanently</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()