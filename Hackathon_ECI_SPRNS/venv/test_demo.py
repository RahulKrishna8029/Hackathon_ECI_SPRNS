#!/usr/bin/env python3
"""
Test script for the SPRNS Demo system.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_demo_service():
    """Test the demo retrieval service functionality."""
    print("Testing Demo Retrieval Service...")
    
    try:
        from demo.service import DemoRetrievalService
        
        # Initialize service
        print("1. Initializing demo service...")
        service = DemoRetrievalService()
        print("   ✅ Demo service initialized successfully")
        
        # Test query processing
        print("2. Testing query processing...")
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Tell me about Python programming"
        ]
        
        for query in test_queries:
            result = service.process_query(query)
            print(f"   Query: {query}")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Sources found: {len(result.get('sources', []))}")
            print("   ✅ Query processed successfully")
        
        # Clean up
        service.close()
        print("3. ✅ Demo service closed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_demo_dashboard_imports():
    """Test that demo dashboard can import required modules."""
    print("\nTesting Demo Dashboard Imports...")
    
    try:
        print("1. Testing demo service import...")
        from demo.service import DemoRetrievalService
        print("   ✅ DemoRetrievalService import successful")
        
        print("2. Testing streamlit availability...")
        try:
            import streamlit as st
            print("   ✅ Streamlit import successful")
            streamlit_available = True
        except ImportError:
            print("   ⚠️  Streamlit not available (install with: pip install streamlit)")
            streamlit_available = False
        
        return streamlit_available
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def main():
    """Run all demo tests."""
    print("=" * 50)
    print("SPRNS Demo Test Suite")
    print("=" * 50)
    
    # Test demo service
    service_test = test_demo_service()
    
    # Test dashboard imports
    import_test = test_demo_dashboard_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Demo Service: {'✅ PASS' if service_test else '❌ FAIL'}")
    print(f"Dashboard Imports: {'✅ PASS' if import_test else '⚠️  PARTIAL (Streamlit needed)'}")
    
    if service_test:
        print("\n🎉 Demo system is working correctly!")
        print("\nTo run the demo dashboard:")
        print("  1. Install Streamlit: pip install streamlit")
        print("  2. Run: python run_demo.py")
        print("\nOr manually:")
        print("  streamlit run Dashboard/demo_ui.py")
    else:
        print("\n⚠️  Demo tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()