#!/usr/bin/env python3
"""
Demo script showing the email analysis capabilities
"""

import os
from email_manager import EmailManager

def demo_without_credentials():
    """Demo showing the application structure without actual email connection"""
    print("🚀 Email Analysis Without Vector DB - Demo")
    print("=" * 60)
    
    print("\n📧 Application Features:")
    print("1. Real-time email fetching and classification")
    print("2. Natural language search and queries")
    print("3. Email analytics and statistics")
    print("4. Chat interface for email questions")
    print("5. Export functionality")
    
    print("\n🏗️ Architecture Components:")
    print("• EmailConnector: IMAP email fetching")
    print("• EmailClassifier: LLM-based classification")
    print("• EmailManager: Orchestrates operations")
    print("• Streamlit App: Web interface")
    print("• CLI Demo: Command-line interface")
    
    print("\n🎯 Email Categories:")
    categories = [
        "inbox", "spam", "junk", "sent", "important",
        "personal", "work", "promotional", "social", "financial"
    ]
    for i, category in enumerate(categories, 1):
        print(f"{i:2d}. {category}")
    
    print("\n🔍 Example Natural Language Queries:")
    queries = [
        "Show me important emails from last week",
        "Find all emails from my boss",
        "How many promotional emails did I receive?",
        "What are the main topics in recent emails?",
        "Are there any urgent emails I should read?",
        "Who sent me the most emails this month?",
        "Find emails about meetings or appointments"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"{i}. \"{query}\"")
    
    print("\n💬 Example Chat Questions:")
    chat_questions = [
        "How many emails did I receive today?",
        "What's the most common email category?",
        "Who are my top email contacts?",
        "Are there any emails I should prioritize?",
        "What promotional offers did I receive?",
        "Show me a summary of today's emails"
    ]
    
    for i, question in enumerate(chat_questions, 1):
        print(f"{i}. \"{question}\"")
    
    print("\n📊 Analytics Features:")
    print("• Email count by category (pie chart)")
    print("• Top senders analysis (bar chart)")
    print("• Category distribution statistics")
    print("• Sender frequency analysis")
    print("• Time-based email patterns")
    
    print("\n🚀 Getting Started:")
    print("1. Set up your .env file with credentials:")
    print("   EMAIL_ADDRESS=your_email@gmail.com")
    print("   EMAIL_PASSWORD=your_app_password")
    print("   OPENAI_API_KEY=your_openai_key")
    print()
    print("2. Run the web application:")
    print("   streamlit run app.py --server.port 12000 --server.address 0.0.0.0")
    print()
    print("3. Or try the CLI version:")
    print("   python cli_demo.py")
    print()
    print("4. Access the web app at:")
    print("   http://localhost:12000")
    
    print("\n✨ Key Benefits:")
    print("• No vector database required - faster processing")
    print("• Real-time email analysis")
    print("• Natural language interface")
    print("• Comprehensive email insights")
    print("• Easy setup and deployment")
    print("• Multiple interface options")
    
    print("\n🔧 Technical Details:")
    print("• Uses OpenAI GPT for classification")
    print("• IMAP protocol for email fetching")
    print("• Streamlit for web interface")
    print("• Pandas for data processing")
    print("• Plotly for visualizations")
    print("• Smart caching for performance")

def demo_email_classification_logic():
    """Demo showing how email classification works"""
    print("\n" + "=" * 60)
    print("🤖 Email Classification Logic Demo")
    print("=" * 60)
    
    # Sample email data
    sample_emails = [
        {
            "subject": "Meeting Tomorrow at 2 PM",
            "sender": "boss@company.com",
            "body": "Hi, we have an important meeting tomorrow at 2 PM to discuss the quarterly results.",
            "expected_category": "work/important"
        },
        {
            "subject": "50% Off Sale - Limited Time!",
            "sender": "noreply@store.com",
            "body": "Don't miss our biggest sale of the year! 50% off everything. Shop now!",
            "expected_category": "promotional"
        },
        {
            "subject": "Your Bank Statement is Ready",
            "sender": "statements@bank.com",
            "body": "Your monthly bank statement for November 2024 is now available online.",
            "expected_category": "financial"
        },
        {
            "subject": "Happy Birthday!",
            "sender": "mom@family.com",
            "body": "Happy birthday sweetie! Hope you have a wonderful day. Love you!",
            "expected_category": "personal"
        }
    ]
    
    print("\n📧 Sample Email Classification:")
    
    for i, email in enumerate(sample_emails, 1):
        print(f"\n{i}. Email Sample:")
        print(f"   Subject: {email['subject']}")
        print(f"   From: {email['sender']}")
        print(f"   Body: {email['body'][:60]}...")
        print(f"   Expected Category: {email['expected_category']}")
        print(f"   Classification Process:")
        print(f"   • Analyze subject keywords")
        print(f"   • Check sender domain/pattern")
        print(f"   • Process email content")
        print(f"   • Apply LLM reasoning")
        print(f"   • Assign confidence score")

def demo_search_capabilities():
    """Demo showing search capabilities"""
    print("\n" + "=" * 60)
    print("🔍 Search Capabilities Demo")
    print("=" * 60)
    
    search_examples = [
        {
            "query": "show me important emails from last week",
            "filters": ["category: important", "date: last 7 days"],
            "description": "Filters by importance and recent timeframe"
        },
        {
            "query": "find promotional emails about sales",
            "filters": ["category: promotional", "content: sale, discount, offer"],
            "description": "Combines category and content keyword filtering"
        },
        {
            "query": "emails from my manager about meetings",
            "filters": ["sender: manager domain", "content: meeting, schedule"],
            "description": "Sender-based filtering with content analysis"
        }
    ]
    
    for i, example in enumerate(search_examples, 1):
        print(f"\n{i}. Search Query: \"{example['query']}\"")
        print(f"   Applied Filters:")
        for filter_item in example['filters']:
            print(f"   • {filter_item}")
        print(f"   Description: {example['description']}")

if __name__ == "__main__":
    demo_without_credentials()
    demo_email_classification_logic()
    demo_search_capabilities()
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("Ready to analyze your emails without vector databases!")
    print("=" * 60)