#!/usr/bin/env python3
"""
CLI Demo for Email Analysis Without Vector DB
"""

import os
import sys
from dotenv import load_dotenv
from email_manager import EmailManager
import json

def main():
    # Load environment variables
    load_dotenv()
    
    print("🔧 Email Analysis CLI Demo")
    print("=" * 50)
    
    # Get credentials
    email_address = os.getenv("EMAIL_ADDRESS") or input("Enter your email address: ")
    email_password = os.getenv("EMAIL_PASSWORD") or input("Enter your email password/app password: ")
    openai_api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
    
    if not all([email_address, email_password, openai_api_key]):
        print("❌ Missing required credentials!")
        return
    
    # Initialize email manager
    print("\n🔌 Connecting to email server...")
    email_manager = EmailManager(email_address, email_password, openai_api_key)
    
    if not email_manager.connect():
        print("❌ Failed to connect to email server!")
        return
    
    print("✅ Connected successfully!")
    
    try:
        while True:
            print("\n" + "=" * 50)
            print("📧 Email Analysis Menu")
            print("1. Fetch and classify emails")
            print("2. Search emails by category")
            print("3. Natural language search")
            print("4. Ask questions about emails")
            print("5. View email statistics")
            print("6. List available folders")
            print("7. Export emails")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                fetch_and_classify_emails(email_manager)
            elif choice == "2":
                search_by_category(email_manager)
            elif choice == "3":
                natural_language_search(email_manager)
            elif choice == "4":
                ask_questions(email_manager)
            elif choice == "5":
                view_statistics(email_manager)
            elif choice == "6":
                list_folders(email_manager)
            elif choice == "7":
                export_emails(email_manager)
            else:
                print("❌ Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    finally:
        email_manager.disconnect()

def fetch_and_classify_emails(email_manager):
    """Fetch and classify emails"""
    folder = input("Enter folder name (default: INBOX): ").strip() or "INBOX"
    limit = int(input("Enter number of emails to fetch (default: 20): ").strip() or "20")
    
    print(f"\n📥 Fetching {limit} emails from {folder}...")
    emails = email_manager.fetch_and_classify_emails(folder=folder, limit=limit)
    
    if not emails:
        print("❌ No emails found!")
        return
    
    print(f"\n✅ Found {len(emails)} emails:")
    print("-" * 80)
    
    for i, email in enumerate(emails[:5], 1):  # Show first 5
        classification = email.get("classification", {})
        print(f"{i}. Subject: {email.get('subject', 'No Subject')[:50]}...")
        print(f"   From: {email.get('sender', 'Unknown')}")
        print(f"   Category: {classification.get('primary_category', 'unknown')} "
              f"(Confidence: {classification.get('confidence', 0):.2f})")
        print(f"   Date: {email.get('date', 'Unknown')}")
        print("-" * 80)
    
    if len(emails) > 5:
        print(f"... and {len(emails) - 5} more emails")

def search_by_category(email_manager):
    """Search emails by category"""
    categories = ["inbox", "spam", "junk", "sent", "important", "personal", "work", "promotional", "social", "financial"]
    
    print("\nAvailable categories:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
    
    try:
        choice = int(input("Enter category number: ")) - 1
        if 0 <= choice < len(categories):
            category = categories[choice]
            folder = input("Enter folder name (default: INBOX): ").strip() or "INBOX"
            
            print(f"\n🔍 Searching for {category} emails in {folder}...")
            emails = email_manager.get_emails_by_category(category, folder=folder)
            
            if emails:
                print(f"\n✅ Found {len(emails)} {category} emails:")
                for i, email in enumerate(emails[:3], 1):
                    print(f"{i}. {email.get('subject', 'No Subject')[:60]}...")
                    print(f"   From: {email.get('sender', 'Unknown')}")
            else:
                print(f"❌ No {category} emails found!")
        else:
            print("❌ Invalid category number!")
    except ValueError:
        print("❌ Please enter a valid number!")

def natural_language_search(email_manager):
    """Natural language search"""
    query = input("Enter your search query (e.g., 'show me important emails from last week'): ").strip()
    
    if not query:
        print("❌ Please enter a search query!")
        return
    
    folder = input("Enter folder name (default: INBOX): ").strip() or "INBOX"
    
    print(f"\n🔍 Searching for: '{query}'...")
    emails = email_manager.search_emails_by_query(query, folder=folder)
    
    if emails:
        print(f"\n✅ Found {len(emails)} matching emails:")
        for i, email in enumerate(emails[:3], 1):
            print(f"{i}. {email.get('subject', 'No Subject')[:60]}...")
            print(f"   From: {email.get('sender', 'Unknown')}")
            print(f"   Category: {email.get('classification', {}).get('primary_category', 'unknown')}")
    else:
        print("❌ No matching emails found!")

def ask_questions(email_manager):
    """Ask questions about emails"""
    question = input("Ask a question about your emails: ").strip()
    
    if not question:
        print("❌ Please enter a question!")
        return
    
    folder = input("Enter folder name (default: INBOX): ").strip() or "INBOX"
    
    print(f"\n🤖 Analyzing emails to answer: '{question}'...")
    answer = email_manager.answer_email_query(question, folder=folder)
    
    print(f"\n💬 Answer:")
    print(answer)

def view_statistics(email_manager):
    """View email statistics"""
    folder = input("Enter folder name (default: INBOX): ").strip() or "INBOX"
    
    print(f"\n📊 Getting statistics for {folder}...")
    stats = email_manager.get_email_statistics(folder=folder)
    
    print(f"\n📈 Email Statistics for {folder}:")
    print(f"Total emails: {stats['total']}")
    
    if stats['categories']:
        print("\nCategory breakdown:")
        for category, count in stats['categories'].items():
            percentage = (count / stats['total']) * 100
            print(f"  • {category}: {count} ({percentage:.1f}%)")
    
    if stats['top_senders']:
        print("\nTop senders:")
        for sender, count in list(stats['top_senders'].items())[:5]:
            print(f"  • {sender}: {count} emails")

def list_folders(email_manager):
    """List available folders"""
    print("\n📁 Available folders:")
    folders = email_manager.get_available_folders()
    
    if folders:
        for i, folder in enumerate(folders, 1):
            print(f"{i}. {folder}")
    else:
        print("❌ No folders found!")

def export_emails(email_manager):
    """Export emails"""
    folder = input("Enter folder name (default: INBOX): ").strip() or "INBOX"
    limit = int(input("Enter number of emails to export (default: 50): ").strip() or "50")
    
    print(f"\n📤 Fetching {limit} emails from {folder} for export...")
    emails = email_manager.fetch_and_classify_emails(folder=folder, limit=limit)
    
    if emails:
        filename = email_manager.export_emails(emails)
        if filename:
            print(f"✅ Exported {len(emails)} emails to {filename}")
        else:
            print("❌ Export failed!")
    else:
        print("❌ No emails to export!")

if __name__ == "__main__":
    main()