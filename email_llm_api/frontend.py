"""
Frontend for Email RAG API using Streamlit
A web interface that communicates with the FastAPI backend
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="📧 Email RAG Frontend",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class EmailRAGClient:
    """Client for communicating with Email RAG API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def connect_email(self, email_config: Dict) -> Dict:
        """Connect to email account"""
        try:
            response = requests.post(
                f"{self.base_url}/connect",
                json=email_config,
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def chat_with_emails(self, message: str, session_id: str) -> Dict:
        """Send chat message to RAG system"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={"message": message, "session_id": session_id},
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Chat failed: {str(e)}"}
    
    def get_stats(self, session_id: str) -> Dict:
        """Get email statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/stats/{session_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to get stats: {str(e)}"}
    
    def get_sessions(self) -> Dict:
        """Get active sessions"""
        try:
            response = requests.get(
                f"{self.base_url}/sessions",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to get sessions: {str(e)}"}
    
    def delete_session(self, session_id: str) -> Dict:
        """Delete a session"""
        try:
            response = requests.delete(
                f"{self.base_url}/sessions/{session_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to delete session: {str(e)}"}
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions"""
        try:
            response = requests.get(f"{self.base_url}/sample-questions")
            response.raise_for_status()
            return response.json().get("questions", [])
        except requests.exceptions.RequestException:
            return [
                "What are the most important emails from this week?",
                "Who sent me the most emails?",
                "Are there any urgent emails I should respond to?"
            ]
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {str(e)}"}

def main():
    st.title("📧 Email RAG Frontend")
    st.markdown("**Connect to your email and chat with your inbox using AI via FastAPI backend**")
    st.markdown("---")
    
    # Initialize session state
    if "client" not in st.session_state:
        st.session_state.client = None
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "connected" not in st.session_state:
        st.session_state.connected = False
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "email_stats" not in st.session_state:
        st.session_state.email_stats = None
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Configuration
    st.sidebar.subheader("🔧 API Settings")
    api_url = st.sidebar.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="URL of the FastAPI backend"
    )
    
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
    
    # Initialize client
    if not st.session_state.client or st.session_state.client.base_url != api_url:
        st.session_state.client = EmailRAGClient(api_url, openai_api_key)
    
    # Health check
    with st.sidebar:
        if st.button("🔍 Check API Health"):
            health = st.session_state.client.health_check()
            if "error" in health:
                st.error(f"❌ API Unhealthy: {health['error']}")
            else:
                st.success(f"✅ API Healthy - {health.get('active_sessions', 0)} active sessions")
    
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
    folder = st.sidebar.selectbox("Email Folder", ["INBOX", "Sent", "Drafts"])
    
    # Connect to email
    if st.sidebar.button("🔗 Connect to Email", type="primary"):
        if not email_address or not email_password:
            st.error("Please enter both email address and password.")
            return
        
        email_config = {
            "email_address": email_address,
            "password": email_password,
            "imap_server": imap_server,
            "port": imap_port,
            "folder": folder,
            "limit": email_limit,
            "days_back": days_back
        }
        
        with st.spinner("Connecting to email and creating RAG system..."):
            result = st.session_state.client.connect_email(email_config)
        
        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.session_state.connected = True
            st.session_state.session_id = result["session_id"]
            st.session_state.email_stats = result.get("stats")
            st.success(f"✅ {result['message']}")
            st.rerun()
    
    # Session management
    if st.session_state.connected:
        st.sidebar.subheader("📋 Session Management")
        st.sidebar.success(f"Session ID: {st.session_state.session_id[:8]}...")
        
        if st.sidebar.button("🗑️ Delete Session"):
            result = st.session_state.client.delete_session(st.session_state.session_id)
            if "error" not in result:
                st.session_state.connected = False
                st.session_state.session_id = None
                st.session_state.chat_history = []
                st.session_state.email_stats = None
                st.success("Session deleted successfully!")
                st.rerun()
    
    # Main content
    if st.session_state.connected and st.session_state.email_stats:
        # Display email statistics
        stats = st.session_state.email_stats
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", stats["total_emails"])
        with col2:
            st.metric("Unique Senders", stats["unique_senders"])
        with col3:
            st.metric("Avg Email Length", f"{stats['avg_length']:.0f} chars")
        with col4:
            st.metric("Date Range", f"{stats['date_range_days']} days")
        
        # Email analysis charts
        if stats["top_senders"]:
            st.subheader("📊 Email Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top senders chart
                senders_df = pd.DataFrame(stats["top_senders"])
                if not senders_df.empty:
                    fig = px.bar(
                        senders_df.head(10), 
                        x='count', 
                        y='sender',
                        orientation='h',
                        title="Top 10 Senders"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Email distribution pie chart
                if len(stats["top_senders"]) > 1:
                    top_5_senders = stats["top_senders"][:5]
                    others_count = sum(sender["count"] for sender in stats["top_senders"][5:])
                    
                    if others_count > 0:
                        top_5_senders.append({"sender": "Others", "count": others_count})
                    
                    pie_df = pd.DataFrame(top_5_senders)
                    fig = px.pie(
                        pie_df, 
                        values='count', 
                        names='sender',
                        title="Email Distribution by Sender"
                    )
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
                            st.write(f"Subject: {source.get('subject', 'N/A')}")
                            st.write(f"From: {source.get('sender', 'N/A')}")
                            st.write(f"Date: {source.get('date', 'N/A')}")
                            if source.get('content_preview'):
                                st.write(f"Preview: {source['content_preview']}")
                            st.write("---")
        
        # Chat input
        if prompt := st.chat_input("Ask about your emails..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from API
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.client.chat_with_emails(
                        prompt, st.session_state.session_id
                    )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(result["answer"])
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result.get("sources", [])
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Show source documents
                    if result.get("sources"):
                        with st.expander("📎 Source Emails"):
                            for i, source in enumerate(result["sources"]):
                                st.write(f"**Email {i+1}:**")
                                st.write(f"Subject: {source.get('subject', 'N/A')}")
                                st.write(f"From: {source.get('sender', 'N/A')}")
                                st.write(f"Date: {source.get('date', 'N/A')}")
                                if source.get('content_preview'):
                                    st.write(f"Preview: {source['content_preview']}")
                                st.write("---")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = st.session_state.client.get_sample_questions()
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions[:6]):  # Show first 6 questions
            col = cols[i % 2]
            if col.button(question, key=f"sample_{i}"):
                # Trigger the question
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.rerun()
    
    elif st.session_state.connected:
        st.info("🔄 Loading email statistics...")
        
        # Try to get stats
        if st.session_state.session_id:
            stats_result = st.session_state.client.get_stats(st.session_state.session_id)
            if "error" not in stats_result:
                st.session_state.email_stats = stats_result
                st.rerun()
            else:
                st.error(f"Failed to load stats: {stats_result['error']}")
    
    else:
        # Instructions
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Enter your OpenAI API Key** in the sidebar
        2. **Configure API URL** (default: http://localhost:8000)
        3. **Set up your email credentials** in the sidebar
        4. **Click 'Connect to Email'** to fetch your emails
        5. **Start chatting** with your email data!
        
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
        
        ### 🔧 API Backend:
        
        Make sure the FastAPI backend is running:
        ```bash
        cd email_llm_api
        pip install -r requirements.txt
        python app.py
        ```
        """)
        
        st.info("💡 **Tip:** This frontend communicates with a FastAPI backend that processes your emails securely!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
        <p>📧 Email RAG Frontend | Built with Streamlit & FastAPI</p>
        <p>Your emails are processed securely via the API backend</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()