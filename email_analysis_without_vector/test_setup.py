#!/usr/bin/env python3
"""
Test script to verify the email analysis setup
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'streamlit',
        'email',
        'imaplib',
        'ssl',
        'openai',
        'pandas',
        'plotly',
        'json',
        'os',
        'datetime',
        're',
        'logging'
    ]
    
    print("🧪 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_local_modules():
    """Test if local modules can be imported"""
    print("\n🧪 Testing local modules...")
    
    local_modules = [
        'email_connector',
        'email_classifier', 
        'email_manager'
    ]
    
    failed_modules = []
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import local modules: {', '.join(failed_modules)}")
        return False
    else:
        print("\n✅ All local modules imported successfully!")
        return True

def test_environment():
    """Test environment setup"""
    print("\n🧪 Testing environment...")
    
    import os
    from dotenv import load_dotenv
    
    # Try to load .env file
    load_dotenv()
    
    required_env_vars = ['EMAIL_ADDRESS', 'EMAIL_PASSWORD', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var} is set")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        return False
    else:
        print("\n✅ All required environment variables are set!")
        return True

def test_email_connection():
    """Test email connection (optional)"""
    print("\n🧪 Testing email connection (optional)...")
    
    try:
        from email_connector import EmailConnector
        import os
        
        email_address = os.getenv('EMAIL_ADDRESS')
        email_password = os.getenv('EMAIL_PASSWORD')
        
        if not email_address or not email_password:
            print("⚠️  Skipping email connection test - credentials not set")
            return True
        
        print(f"Attempting to connect to {email_address}...")
        connector = EmailConnector(email_address, email_password)
        
        if connector.connect():
            print("✅ Email connection successful!")
            connector.disconnect()
            return True
        else:
            print("❌ Email connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Email connection test error: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection (optional)"""
    print("\n🧪 Testing OpenAI API connection (optional)...")
    
    try:
        import openai
        import os
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("⚠️  Skipping OpenAI test - API key not set")
            return True
        
        client = openai.OpenAI(api_key=api_key)
        
        # Simple test request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ OpenAI API connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Email Analysis Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Local Modules", test_local_modules),
        ("Environment Variables", test_environment),
        ("Email Connection", test_email_connection),
        ("OpenAI API", test_openai_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nYou can now run:")
        print("  • Web app: streamlit run app.py")
        print("  • CLI demo: python cli_demo.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure to:")
        print("  • Install all requirements: pip install -r requirements.txt")
        print("  • Set up your .env file with credentials")
        print("  • Check your internet connection")

if __name__ == "__main__":
    main()