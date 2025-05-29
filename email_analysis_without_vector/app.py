import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from email_manager import EmailManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Email Analysis Without Vector DB",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .email-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'email_manager' not in st.session_state:
        st.session_state.email_manager = None
    if 'is_connected' not in st.session_state:
        st.session_state.is_connected = False
    if 'current_emails' not in st.session_state:
        st.session_state.current_emails = []
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = "INBOX"

def get_category_color(category):
    """Get color for category badge"""
    colors = {
        "inbox": "#007bff",
        "spam": "#dc3545",
        "junk": "#ffc107",
        "sent": "#28a745",
        "important": "#fd7e14",
        "personal": "#6f42c1",
        "work": "#20c997",
        "promotional": "#e83e8c",
        "social": "#17a2b8",
        "financial": "#6c757d"
    }
    return colors.get(category, "#6c757d")

def display_email_card(email, index):
    """Display an email in a card format"""
    classification = email.get("classification", {})
    category = classification.get("primary_category", "unknown")
    confidence = classification.get("confidence", 0)
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="email-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; color: #333;">{email.get('subject', 'No Subject')}</h4>
                    <span class="category-badge" style="background-color: {get_category_color(category)}; color: white;">
                        {category.upper()}
                    </span>
                </div>
                <p style="margin: 0.25rem 0; color: #666;"><strong>From:</strong> {email.get('sender', 'Unknown')}</p>
                <p style="margin: 0.25rem 0; color: #666;"><strong>Date:</strong> {email.get('date', 'Unknown')}</p>
                <p style="margin: 0.5rem 0; color: #333;">{email.get('body', '')[:200]}...</p>
                <small style="color: #888;">Confidence: {confidence:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button(f"View Details", key=f"view_{index}"):
                st.session_state[f"show_details_{index}"] = not st.session_state.get(f"show_details_{index}", False)
        
        # Show details if button clicked
        if st.session_state.get(f"show_details_{index}", False):
            st.markdown("**Full Email Body:**")
            st.text_area("", value=email.get('body', ''), height=200, key=f"body_{index}")
            
            st.markdown("**Classification Details:**")
            st.json(classification)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📧 Email Analysis Without Vector DB</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Email credentials
        st.subheader("Email Settings")
        email_address = st.text_input("Email Address", value=os.getenv("EMAIL_ADDRESS", ""))
        email_password = st.text_input("Email Password/App Password", type="password", value=os.getenv("EMAIL_PASSWORD", ""))
        
        # OpenAI API key
        st.subheader("AI Settings")
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        
        # Connection button
        if st.button("🔌 Connect to Email", type="primary"):
            if email_address and email_password and openai_api_key:
                try:
                    with st.spinner("Connecting to email server..."):
                        st.session_state.email_manager = EmailManager(email_address, email_password, openai_api_key)
                        if st.session_state.email_manager.connect():
                            st.session_state.is_connected = True
                            st.success("✅ Connected successfully!")
                        else:
                            st.error("❌ Failed to connect. Please check your credentials.")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
            else:
                st.error("Please fill in all required fields.")
        
        # Disconnect button
        if st.session_state.is_connected:
            if st.button("🔌 Disconnect"):
                st.session_state.email_manager.disconnect()
                st.session_state.is_connected = False
                st.session_state.current_emails = []
                st.success("Disconnected successfully!")
        
        # Connection status
        if st.session_state.is_connected:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Not Connected")
    
    # Main content
    if not st.session_state.is_connected:
        st.info("👈 Please configure your email settings and connect to get started.")
        
        # Instructions
        st.markdown("""
        ## 📋 Setup Instructions
        
        1. **Email Configuration:**
           - Enter your email address
           - For Gmail, use an App Password (not your regular password)
           - Enable 2-factor authentication and generate an App Password
        
        2. **OpenAI API Key:**
           - Get your API key from OpenAI platform
           - This is used for email classification and natural language queries
        
        3. **Features:**
           - 📥 Fetch and classify emails automatically
           - 🔍 Search emails using natural language
           - 📊 View email statistics and analytics
           - 💬 Ask questions about your emails
           - 📁 Browse different email folders
        """)
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Email Browser", "🔍 Search & Query", "📊 Analytics", "💬 Chat", "⚙️ Settings"])
    
    with tab1:
        st.header("📥 Email Browser")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Folder selection
            folders = st.session_state.email_manager.get_available_folders()
            selected_folder = st.selectbox("Select Folder", folders, index=0 if "INBOX" not in folders else folders.index("INBOX"))
            st.session_state.selected_folder = selected_folder
        
        with col2:
            # Email limit
            email_limit = st.number_input("Number of Emails", min_value=10, max_value=200, value=50, step=10)
        
        with col3:
            # Refresh button
            if st.button("🔄 Refresh", type="primary"):
                with st.spinner("Fetching and classifying emails..."):
                    st.session_state.current_emails = st.session_state.email_manager.refresh_cache(
                        folder=selected_folder, limit=email_limit
                    )
        
        # Fetch emails if not already loaded
        if not st.session_state.current_emails:
            with st.spinner("Fetching and classifying emails..."):
                st.session_state.current_emails = st.session_state.email_manager.fetch_and_classify_emails(
                    folder=selected_folder, limit=email_limit
                )
        
        # Display emails
        if st.session_state.current_emails:
            st.subheader(f"📧 {len(st.session_state.current_emails)} Emails from {selected_folder}")
            
            # Category filter
            categories = list(set([email.get("classification", {}).get("primary_category", "unknown") 
                                 for email in st.session_state.current_emails]))
            selected_categories = st.multiselect("Filter by Category", categories, default=categories)
            
            # Filter emails
            filtered_emails = [
                email for email in st.session_state.current_emails
                if email.get("classification", {}).get("primary_category") in selected_categories
            ]
            
            # Display filtered emails
            for i, email in enumerate(filtered_emails):
                display_email_card(email, i)
        else:
            st.info("No emails found in the selected folder.")
    
    with tab2:
        st.header("🔍 Search & Query")
        
        # Natural language search
        st.subheader("Natural Language Search")
        search_query = st.text_input("Search emails (e.g., 'show me important emails from last week', 'find promotional emails')")
        
        if st.button("🔍 Search") and search_query:
            with st.spinner("Searching emails..."):
                search_results = st.session_state.email_manager.search_emails_by_query(
                    search_query, folder=st.session_state.selected_folder
                )
            
            if search_results:
                st.subheader(f"📧 {len(search_results)} Search Results")
                for i, email in enumerate(search_results):
                    display_email_card(email, f"search_{i}")
            else:
                st.info("No emails found matching your search criteria.")
        
        # Category-based filtering
        st.subheader("Filter by Category")
        col1, col2 = st.columns(2)
        
        with col1:
            category_filter = st.selectbox("Select Category", [
                "inbox", "spam", "junk", "sent", "important", 
                "personal", "work", "promotional", "social", "financial"
            ])
        
        with col2:
            if st.button("📁 Filter by Category"):
                with st.spinner("Filtering emails..."):
                    category_emails = st.session_state.email_manager.get_emails_by_category(
                        category_filter, folder=st.session_state.selected_folder
                    )
                
                if category_emails:
                    st.subheader(f"📧 {len(category_emails)} {category_filter.title()} Emails")
                    for i, email in enumerate(category_emails):
                        display_email_card(email, f"cat_{i}")
                else:
                    st.info(f"No {category_filter} emails found.")
    
    with tab3:
        st.header("📊 Email Analytics")
        
        if st.session_state.current_emails:
            # Get statistics
            stats = st.session_state.email_manager.get_email_statistics(
                folder=st.session_state.selected_folder
            )
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Emails", stats["total"])
            
            with col2:
                st.metric("Categories", len(stats["categories"]))
            
            with col3:
                st.metric("Unique Senders", len(stats["top_senders"]))
            
            with col4:
                most_common_category = max(stats["categories"], key=stats["categories"].get) if stats["categories"] else "N/A"
                st.metric("Top Category", most_common_category)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                if stats["categories"]:
                    fig_pie = px.pie(
                        values=list(stats["categories"].values()),
                        names=list(stats["categories"].keys()),
                        title="Email Distribution by Category"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Top senders
                if stats["top_senders"]:
                    fig_bar = px.bar(
                        x=list(stats["top_senders"].values()),
                        y=list(stats["top_senders"].keys()),
                        orientation='h',
                        title="Top Email Senders"
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Detailed statistics
            st.subheader("📈 Detailed Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Category Breakdown:**")
                for category, count in stats["categories"].items():
                    percentage = (count / stats["total"]) * 100
                    st.write(f"• {category.title()}: {count} ({percentage:.1f}%)")
            
            with col2:
                st.markdown("**Top Senders:**")
                for sender, count in list(stats["top_senders"].items())[:5]:
                    st.write(f"• {sender}: {count} emails")
        else:
            st.info("Load emails first to view analytics.")
    
    with tab4:
        st.header("💬 Chat with Your Emails")
        
        st.markdown("Ask questions about your emails in natural language!")
        
        # Chat interface
        query = st.text_area("Ask a question about your emails:", 
                           placeholder="e.g., 'How many important emails did I receive today?', 'Who sent me the most emails?', 'What are the main topics in my recent emails?'")
        
        if st.button("💬 Ask") and query:
            if st.session_state.current_emails:
                with st.spinner("Analyzing your emails..."):
                    answer = st.session_state.email_manager.answer_email_query(
                        query, folder=st.session_state.selected_folder
                    )
                
                st.subheader("🤖 Answer:")
                st.write(answer)
            else:
                st.warning("Please load emails first before asking questions.")
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "How many emails did I receive today?",
            "Show me all emails from my boss",
            "What are the most common email categories?",
            "Which emails are marked as important?",
            "Who are my top email contacts?",
            "What promotional emails did I receive?",
            "Are there any urgent emails I should read?"
        ]
        
        for example in example_queries:
            if st.button(f"💭 {example}", key=f"example_{example}"):
                if st.session_state.current_emails:
                    with st.spinner("Analyzing your emails..."):
                        answer = st.session_state.email_manager.answer_email_query(
                            example, folder=st.session_state.selected_folder
                        )
                    
                    st.subheader("🤖 Answer:")
                    st.write(answer)
                else:
                    st.warning("Please load emails first.")
    
    with tab5:
        st.header("⚙️ Settings & Export")
        
        # Export functionality
        st.subheader("📤 Export Emails")
        
        if st.session_state.current_emails:
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Export Format", ["JSON", "CSV"])
            
            with col2:
                if st.button("📤 Export Current Emails"):
                    if export_format == "JSON":
                        filename = st.session_state.email_manager.export_emails(st.session_state.current_emails)
                        if filename:
                            st.success(f"✅ Emails exported to {filename}")
                    else:  # CSV
                        # Convert to DataFrame for CSV export
                        df_data = []
                        for email in st.session_state.current_emails:
                            classification = email.get("classification", {})
                            df_data.append({
                                "Subject": email.get("subject", ""),
                                "Sender": email.get("sender", ""),
                                "Date": email.get("date", ""),
                                "Category": classification.get("primary_category", ""),
                                "Confidence": classification.get("confidence", 0),
                                "Body_Preview": email.get("body", "")[:200]
                            })
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"emails_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        else:
            st.info("Load emails first to enable export functionality.")
        
        # Cache management
        st.subheader("🗄️ Cache Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache"):
                if hasattr(st.session_state.email_manager, 'cached_emails'):
                    st.session_state.email_manager.cached_emails.clear()
                st.session_state.current_emails = []
                st.success("Cache cleared successfully!")
        
        with col2:
            cache_size = len(st.session_state.email_manager.cached_emails) if st.session_state.email_manager else 0
            st.metric("Cached Folders", cache_size)
        
        # Application info
        st.subheader("ℹ️ Application Info")
        st.markdown("""
        **Email Analysis Without Vector DB**
        
        This application provides:
        - Real-time email fetching and classification
        - Natural language search and queries
        - Email analytics and statistics
        - No vector database dependency for faster processing
        - Direct LLM-based classification and analysis
        
        **Note:** This app processes emails in real-time without storing them in a vector database, 
        which makes it faster for fresh email analysis but requires re-processing for each session.
        """)

if __name__ == "__main__":
    main()