#!/usr/bin/env python3
"""
Test script to verify the integration between retrieval service and dashboard.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_retrieval_service():
    """Test the retrieval service functionality."""
    print("Testing Retrieval Service Integration...")
    
    try:
        from retrieval.retrieval_service import RetrievalService
        
        # Initialize service
        print("1. Initializing retrieval service...")
        service = RetrievalService()
        print("   ✅ Service initialized successfully")
        
        # Test query processing
        print("2. Testing query processing...")
        test_query = "What is machine learning?"
        result = service.process_query(test_query)
        
        print(f"   Query: {test_query}")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Answer: {result.get('answer', 'No answer')[:100]}...")
        print(f"   Sources found: {len(result.get('sources', []))}")
        print("   ✅ Query processing successful")
        
        # Clean up
        service.close()
        print("3. ✅ Service closed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_dashboard_imports():
    """Test that dashboard can import required modules."""
    print("\nTesting Dashboard Imports...")
    
    try:
        # Test importing the dashboard module
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dashboard"))
        
        print("1. Testing retrieval service import...")
        from retrieval.retrieval_service import RetrievalService
        print("   ✅ RetrievalService import successful")
        
        print("2. Testing streamlit availability...")
        import streamlit as st
        print("   ✅ Streamlit import successful")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("SPRNS Integration Test Suite")
    print("=" * 50)
    
    # Test retrieval service
    service_test = test_retrieval_service()
    
    # Test dashboard imports
    import_test = test_dashboard_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Retrieval Service: {'✅ PASS' if service_test else '❌ FAIL'}")
    print(f"Dashboard Imports: {'✅ PASS' if import_test else '❌ FAIL'}")
    
    if service_test and import_test:
        print("\n🎉 All tests passed! The integration is working correctly.")
        print("\nTo run the dashboard:")
        print("  python run_dashboard.py")
        print("\nOr manually:")
        print("  streamlit run Dashboard/sprns_ui.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()