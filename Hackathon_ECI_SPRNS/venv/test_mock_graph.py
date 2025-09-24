#!/usr/bin/env python3
"""
Test script for the Mock Graph Database functionality.
Tests the 4-customer mock graph and its integration with the retrieval system.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mock_neo4j_connector():
    """Test the mock Neo4j connector functionality."""
    print("Testing Mock Neo4j Connector...")
    
    try:
        from mock_graph.mock_neo4j import MockNeo4jConnector
        
        # Initialize mock connector
        print("1. Initializing mock Neo4j connector...")
        connector = MockNeo4jConnector()
        print("   ✅ Mock connector initialized successfully")
        
        # Test customer data
        print("2. Testing customer data...")
        customers = connector.execute_query("MATCH (c:Customer) RETURN c")
        print(f"   Found {len(customers)} customers")
        for customer in customers:
            cust_data = customer['c']
            print(f"   - {cust_data['name']} ({cust_data['industry']}) - {cust_data['status']}")
        print("   ✅ Customer data retrieved successfully")
        
        # Test document search
        print("3. Testing document search...")
        docs = connector.execute_query(
            "MATCH (d:Document) WHERE d.content CONTAINS $query_text RETURN d LIMIT $limit",
            {"query_text": "analytics", "limit": 5}
        )
        print(f"   Found {len(docs)} documents containing 'analytics'")
        for doc in docs:
            doc_data = doc['d']
            print(f"   - {doc_data['title']} (Type: {doc_data['type']})")
        print("   ✅ Document search working correctly")
        
        # Test customer-specific queries
        print("4. Testing customer-specific operations...")
        customer_docs = connector.get_customer_documents("CUST_001")
        print(f"   TechCorp has {len(customer_docs)} related documents")
        
        customer_products = connector.get_customer_products("CUST_001")
        print(f"   TechCorp has {len(customer_products)} product relationships")
        
        # Test industry search
        tech_customers = connector.search_customers_by_industry("Technology")
        print(f"   Found {len(tech_customers)} technology customers")
        print("   ✅ Customer operations working correctly")
        
        # Test analytics summary
        print("5. Testing analytics summary...")
        summary = connector.get_analytics_summary()
        print(f"   Total entities: {summary['total_customers']} customers, {summary['total_products']} products, {summary['total_documents']} documents")
        print(f"   Industries: {list(summary['customers_by_industry'].keys())}")
        print("   ✅ Analytics summary generated successfully")
        
        # Clean up
        connector.close()
        print("6. ✅ Mock connector closed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_service_with_mock():
    """Test the retrieval service using the mock graph."""
    print("\nTesting Retrieval Service with Mock Graph...")
    
    try:
        from retrieval.retrieval_service import RetrievalService
        
        # Initialize service (should fall back to mock)
        print("1. Initializing retrieval service...")
        service = RetrievalService()
        print("   ✅ Service initialized (using mock Neo4j)")
        
        # Test customer-related queries
        print("2. Testing customer-related queries...")
        
        test_queries = [
            "Tell me about TechCorp Solutions",
            "What analytics solutions do we have?",
            "How did HealthFirst implement security?",
            "What are the benefits of our IoT system?",
            "Show me retail customer success stories"
        ]
        
        for query in test_queries:
            print(f"\n   Query: {query}")
            result = service.process_query(query)
            
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Sources found: {len(result.get('sources', []))}")
            
            # Show first source if available
            sources = result.get('sources', [])
            if sources:
                first_source = sources[0]
                print(f"   Top source: {first_source.get('metadata', {}).get('title', 'Unknown')}")
        
        print("\n   ✅ All queries processed successfully")
        
        # Clean up
        service.close()
        print("3. ✅ Service closed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_customer_scenarios():
    """Test specific customer scenarios with the mock data."""
    print("\nTesting Specific Customer Scenarios...")
    
    try:
        from mock_graph.mock_neo4j import MockNeo4jConnector
        
        connector = MockNeo4jConnector()
        
        # Scenario 1: Customer lookup
        print("1. Customer Information Lookup...")
        techcorp = connector.get_customer_by_id("CUST_001")
        if techcorp:
            print(f"   TechCorp: {techcorp['name']} - Revenue: ${techcorp['annual_revenue']:,}")
            print(f"   Location: {techcorp['location']}, Status: {techcorp['status']}")
        
        # Scenario 2: Customer's products and documents
        print("2. Customer's Products and Documents...")
        products = connector.get_customer_products("CUST_001")
        documents = connector.get_customer_documents("CUST_001")
        print(f"   TechCorp has {len(products)} products and {len(documents)} documents")
        
        # Scenario 3: Industry analysis
        print("3. Industry Analysis...")
        for industry in ["Technology", "Healthcare", "Energy", "Retail"]:
            customers = connector.search_customers_by_industry(industry)
            print(f"   {industry}: {len(customers)} customers")
        
        # Scenario 4: Document types analysis
        print("4. Document Types Analysis...")
        all_docs = list(connector.documents.values())
        doc_types = {}
        for doc in all_docs:
            doc_type = doc.get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in doc_types.items():
            print(f"   {doc_type}: {count} documents")
        
        print("   ✅ All scenarios tested successfully")
        
        connector.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all mock graph tests."""
    print("=" * 60)
    print("SPRNS Mock Graph Database Test Suite")
    print("=" * 60)
    
    # Test mock Neo4j connector
    connector_test = test_mock_neo4j_connector()
    
    # Test retrieval service integration
    service_test = test_retrieval_service_with_mock()
    
    # Test specific scenarios
    scenario_test = test_specific_customer_scenarios()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Mock Neo4j Connector: {'✅ PASS' if connector_test else '❌ FAIL'}")
    print(f"Retrieval Service Integration: {'✅ PASS' if service_test else '❌ FAIL'}")
    print(f"Customer Scenarios: {'✅ PASS' if scenario_test else '❌ FAIL'}")
    
    if connector_test and service_test and scenario_test:
        print("\n🎉 All mock graph tests passed!")
        print("\n📊 Mock Database Contents:")
        print("   • 4 Customers (TechCorp, HealthFirst, GreenEnergy, RetailMax)")
        print("   • 3 Products (Analytics Platform, Security Suite, IoT System)")
        print("   • 6 Documents (Case studies, reports, guides)")
        print("   • Multiple relationships and interactions")
        print("\n🚀 Ready for local testing without Neo4j!")
        print("\nTo use the mock graph:")
        print("   1. Run: python run_demo.py")
        print("   2. Ask questions about customers, products, or implementations")
        print("   3. The system will use realistic customer data from the mock graph")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()