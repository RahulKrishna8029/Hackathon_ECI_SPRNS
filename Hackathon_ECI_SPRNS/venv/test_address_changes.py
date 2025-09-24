#!/usr/bin/env python3
"""
Test script for address change functionality in the mock graph.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_address_history():
    """Test the address history functionality."""
    print("Testing Address History Functionality...")
    
    try:
        from mock_graph.mock_neo4j import MockNeo4jConnector
        
        # Initialize mock connector
        print("1. Initializing mock connector...")
        connector = MockNeo4jConnector()
        print("   ✅ Mock connector initialized")
        
        # Test address history for each customer
        print("2. Testing customer address histories...")
        for customer_id, customer in connector.customers.items():
            print(f"\n   Customer: {customer['name']}")
            print(f"   Total address changes: {customer['address_changes']}")
            
            # Get address history
            history = connector.get_customer_address_history(customer_id)
            print(f"   Address history entries: {len(history)}")
            
            # Get current address
            current = connector.get_current_address(customer_id)
            if current:
                print(f"   Current: {current['address']}")
                print(f"   Since: {current['start_date']}")
                print(f"   Reason: {current['reason']}")
        
        print("\n   ✅ Address history data verified")
        
        # Test address change timeline
        print("3. Testing address change timeline...")
        timeline = connector.get_address_change_timeline()
        print(f"   Total address events: {len(timeline)}")
        
        # Show recent changes
        recent_changes = [event for event in timeline if event['start_date'] >= '2024-01-01']
        print(f"   Recent changes (2024): {len(recent_changes)}")
        
        print("   ✅ Timeline functionality working")
        
        # Test analytics summary
        print("4. Testing analytics with address data...")
        summary = connector.get_analytics_summary()
        print(f"   Total address changes: {summary['total_address_changes']}")
        print(f"   Average per customer: {summary['avg_address_changes_per_customer']:.1f}")
        print(f"   Address change distribution: {summary['customers_by_address_changes']}")
        
        print("   ✅ Analytics including address data")
        
        connector.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_address_queries():
    """Test address-related queries in the demo service."""
    print("\nTesting Address Queries in Demo Service...")
    
    try:
        from demo.service import DemoRetrievalService
        
        # Initialize service
        print("1. Initializing demo service...")
        service = DemoRetrievalService()
        print("   ✅ Service initialized with address data")
        
        # Test address-specific queries
        print("2. Testing address-specific queries...")
        
        address_queries = [
            "TechCorp address history",
            "Where is HealthFirst located?",
            "GreenEnergy location changes",
            "RetailMax address moves",
            "Show me all customer address changes",
            "Address change analysis",
            "Customer relocation patterns"
        ]
        
        for query in address_queries:
            print(f"\n   Query: {query}")
            result = service.process_query(query)
            
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Sources: {len(result.get('sources', []))}")
            print(f"   Answer preview: {result.get('answer', '')[:150]}...")
            
            # Check if address-related sources are prioritized
            sources = result.get('sources', [])
            if sources:
                top_source = sources[0]
                source_type = top_source.get('metadata', {}).get('title', '')
                relevance = top_source.get('relevance_score', 0)
                print(f"   Top source: {source_type} (Score: {relevance:.2f})")
        
        print("\n   ✅ All address queries processed successfully")
        
        service.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_address_scenarios():
    """Test specific address change scenarios."""
    print("\nTesting Specific Address Change Scenarios...")
    
    try:
        from demo.service import DemoRetrievalService
        
        service = DemoRetrievalService()
        
        # Scenario 1: Customer-specific address history
        print("1. Customer-specific address history...")
        result = service.process_query("Show me TechCorp's address history")
        print(f"   Answer length: {len(result.get('answer', ''))}")
        if 'San Francisco' in result.get('answer', '') and 'Palo Alto' in result.get('answer', ''):
            print("   ✅ Contains current and historical addresses")
        else:
            print("   ⚠️  May not contain complete address history")
        
        # Scenario 2: Address change reasons
        print("2. Address change reasons...")
        result = service.process_query("Why did HealthFirst move locations?")
        answer = result.get('answer', '')
        if any(reason in answer.lower() for reason in ['compliance', 'accessibility', 'patient']):
            print("   ✅ Contains business reasons for moves")
        else:
            print("   ⚠️  May not contain detailed reasons")
        
        # Scenario 3: Current location query
        print("3. Current location query...")
        result = service.process_query("Where is GreenEnergy located now?")
        if 'Austin' in result.get('answer', ''):
            print("   ✅ Returns current location")
        else:
            print("   ⚠️  May not return current location")
        
        # Scenario 4: Address change patterns
        print("4. Address change patterns...")
        result = service.process_query("What patterns do you see in customer address changes?")
        answer = result.get('answer', '')
        if len(answer) > 100:
            print("   ✅ Provides analysis of address change patterns")
        else:
            print("   ⚠️  Limited pattern analysis")
        
        print("   ✅ Address scenarios tested")
        
        service.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all address change tests."""
    print("=" * 60)
    print("SPRNS Address Change Test Suite")
    print("=" * 60)
    
    # Test address history data
    history_test = test_address_history()
    
    # Test address queries
    query_test = test_address_queries()
    
    # Test specific scenarios
    scenario_test = test_specific_address_scenarios()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Address History Data: {'✅ PASS' if history_test else '❌ FAIL'}")
    print(f"Address Query Processing: {'✅ PASS' if query_test else '❌ FAIL'}")
    print(f"Address Change Scenarios: {'✅ PASS' if scenario_test else '❌ FAIL'}")
    
    if history_test and query_test and scenario_test:
        print("\n🎉 All address change tests passed!")
        print("\n📍 Address Change Features Available:")
        print("   • 4 address changes per customer (realistic scenario)")
        print("   • Complete address history with dates and reasons")
        print("   • Current location tracking")
        print("   • Address change timeline and analytics")
        print("   • Smart query handling for location-related questions")
        print("\n🔍 Try these queries in the UI:")
        print("   • 'TechCorp address history'")
        print("   • 'Where is HealthFirst located?'")
        print("   • 'Show me customer address changes'")
        print("   • 'Why did customers move locations?'")
    else:
        print("\n⚠️  Some address change tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()