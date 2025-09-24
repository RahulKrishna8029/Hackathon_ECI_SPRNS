#!/usr/bin/env python3
"""
Clean demo service that only uses customer and address data.
Completely removes any generic AI/ML content.
"""

import time
import random
from typing import List, Dict, Any

class CleanDemoRetrievalService:
    """
    Ultra-clean demo service that ONLY uses customer and address data.
    No generic AI/ML content whatsoever.
    """
    
    def __init__(self):
        """Initialize with only customer data."""
        # Initialize mock connector for customer data
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from mock_graph.mock_neo4j import MockNeo4jConnector
            self.mock_connector = MockNeo4jConnector()
            self.knowledge_base = []
            self._load_customer_data_only()
            print("Clean Demo Service initialized with ONLY customer data!")
        except Exception as e:
            self.mock_connector = None
            self.knowledge_base = []
            print(f"Clean Demo Service initialized (basic mode): {e}")
    
    def _load_customer_data_only(self):
        """Load ONLY customer and address data - no generic content."""
        if not self.mock_connector:
            return
        
        # Add customer documents from mock graph
        for doc_id, doc in self.mock_connector.documents.items():
            customer_doc = {
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'category': doc.get('type', 'Customer Data'),
                'keywords': doc.get('tags', []),
                'customer_id': doc.get('customer_id'),
                'created_date': doc.get('created_date')
            }
            self.knowledge_base.append(customer_doc)
        
        # Add customer profiles
        for cust_id, customer in self.mock_connector.customers.items():
            current_address = self.mock_connector.get_current_address(cust_id)
            address_info = f"Currently located at {customer['current_location']}"
            if current_address:
                address_info += f" (moved {customer['address_changes']} times, last change: {customer['last_address_change']})"
            
            customer_profile = {
                'id': f"customer_{cust_id}",
                'title': f"Customer Profile: {customer['name']}",
                'content': f"{customer['name']} is a {customer['size']} {customer['industry']} company. {address_info}. Annual revenue: ${customer['annual_revenue']:,}. Status: {customer['status']}. Contact: {customer['contact_email']}. Last interaction: {customer['last_interaction']}. This customer has changed addresses {customer['address_changes']} times.",
                'category': 'Customer Profile',
                'keywords': [customer['industry'].lower(), customer['size'].lower(), 'customer', 'profile', 'address', 'location'],
                'customer_id': cust_id,
                'created_date': customer.get('created_date'),
                'customer_name': customer['name']
            }
            self.knowledge_base.append(customer_profile)
        
        print(f"Loaded {len(self.knowledge_base)} customer-only documents")
    
    def _calculate_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate relevance score focused on customer data."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Customer name matching
        customer_names = {
            'techcorp': 'CUST_001',
            'tech corp': 'CUST_001', 
            'healthfirst': 'CUST_002',
            'health first': 'CUST_002',
            'greenenergy': 'CUST_003',
            'green energy': 'CUST_003',
            'retailmax': 'CUST_004',
            'retail max': 'CUST_004'
        }
        
        # Strong boost for customer matches
        customer_boost = 0
        for name, cust_id in customer_names.items():
            if name in query_lower and document.get('customer_id') == cust_id:
                customer_boost = 3.0
                break
            elif name in query_lower and document.get('category') == 'Customer Profile':
                customer_boost = 2.0
                break
        
        # Address/location boost
        address_boost = 0
        address_terms = ['address', 'location', 'move', 'moved', 'relocation', 'relocate', 'change', 'history']
        if any(term in query_lower for term in address_terms):
            if document.get('category') == 'address_history' or any(term in document.get('content', '').lower() for term in address_terms):
                address_boost = 2.0
        
        # Basic text matching
        title_words = set(document['title'].lower().split())
        content_words = set(document['content'].lower().split())
        
        title_overlap = len(query_words.intersection(title_words))
        content_overlap = len(query_words.intersection(content_words))
        
        base_score = (title_overlap * 0.4 + content_overlap * 0.4)
        total_score = base_score + customer_boost + address_boost
        
        return min(total_score, 5.0)
    
    def retrieve_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve only customer-related documents."""
        scored_docs = []
        
        for doc in self.knowledge_base:
            relevance = self._calculate_relevance(query, doc)
            if relevance > 0.1:
                doc_with_score = doc.copy()
                doc_with_score['relevance_score'] = relevance
                scored_docs.append(doc_with_score)
        
        # Sort by relevance
        scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_docs[:limit]
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate customer-focused answers only."""
        if not documents:
            return "I couldn't find any customer information for your query. Try asking about TechCorp, HealthFirst, GreenEnergy, or RetailMax."
        
        query_lower = query.lower()
        primary_doc = documents[0]
        
        # Customer-specific responses
        customer_queries = {
            'techcorp': 'TechCorp Solutions',
            'tech corp': 'TechCorp Solutions',
            'healthfirst': 'HealthFirst Medical', 
            'health first': 'HealthFirst Medical',
            'greenenergy': 'GreenEnergy Inc',
            'green energy': 'GreenEnergy Inc',
            'retailmax': 'RetailMax Chain',
            'retail max': 'RetailMax Chain'
        }
        
        # Address queries with specific location targeting
        if any(term in query_lower for term in ['address', 'location', 'move', 'history']):
            for customer_key, customer_name in customer_queries.items():
                if customer_key in query_lower and hasattr(self, 'mock_connector') and self.mock_connector:
                    name_to_id = {
                        'techcorp solutions': 'CUST_001',
                        'healthfirst medical': 'CUST_002', 
                        'greenenergy inc': 'CUST_003',
                        'retailmax chain': 'CUST_004'
                    }
                    customer_id = name_to_id.get(customer_name.lower())
                    
                    if customer_id:
                        address_history = self.mock_connector.get_customer_address_history(customer_id)
                        
                        if address_history:
                            # Check for specific location requests
                            if any(term in query_lower for term in ['second location', 'second address', 'first move', 'moved first']):
                                # Return second location (first move)
                                if len(address_history) >= 2:
                                    second_location = address_history[1]  # Index 1 is the second location
                                    return f"{customer_name} Second Location (First Move):\n\n📍 {second_location['address']}\n📅 Period: {second_location['start_date']} to {second_location['end_date']}\n💡 Reason: {second_location['reason']}\n\nThis was their first relocation from the original location."
                                else:
                                    return f"{customer_name} has not moved to a second location yet."
                            
                            elif any(term in query_lower for term in ['third location', 'third address', 'second move']):
                                # Return third location (second move)
                                if len(address_history) >= 3:
                                    third_location = address_history[2]  # Index 2 is the third location
                                    return f"{customer_name} Third Location (Second Move):\n\n📍 {third_location['address']}\n📅 Period: {third_location['start_date']} to {third_location['end_date']}\n💡 Reason: {third_location['reason']}\n\nThis was their second relocation."
                                else:
                                    return f"{customer_name} has not moved to a third location yet."
                            
                            elif any(term in query_lower for term in ['first location', 'original location', 'initial address']):
                                # Return first/original location
                                first_location = address_history[0]  # Index 0 is the original location
                                return f"{customer_name} Original Location:\n\n📍 {first_location['address']}\n📅 Period: {first_location['start_date']} to {first_location['end_date']}\n💡 Reason: {first_location['reason']}\n\nThis was their founding location."
                            
                            elif any(term in query_lower for term in ['current location', 'current address', 'where now']):
                                # Return current location only
                                current_address = self.mock_connector.get_current_address(customer_id)
                                return f"{customer_name} Current Location:\n\n📍 {current_address['address']}\n📅 Since: {current_address['start_date']}\n💡 Reason: {current_address['reason']}\n\nThis is their current headquarters."
                            
                            else:
                                # Default: Return full history
                                current_address = self.mock_connector.get_current_address(customer_id)
                                response = f"{customer_name} Address History:\n\n"
                                response += f"📍 Current Location: {current_address['address']}\n"
                                response += f"   Since: {current_address['start_date']}\n"
                                response += f"   Reason: {current_address['reason']}\n\n"
                                response += f"📋 Previous Locations ({len(address_history)-1} moves):\n"
                                
                                for i, addr in enumerate(address_history[:-1], 1):
                                    response += f"{i}. {addr['address']}\n"
                                    response += f"   Period: {addr['start_date']} to {addr['end_date']}\n"
                                    response += f"   Reason: {addr['reason']}\n\n"
                                
                                return response
        
        # Customer profile queries
        for customer_key, customer_name in customer_queries.items():
            if customer_key in query_lower:
                customer_docs = [doc for doc in documents if 
                               (doc.get('category') == 'Customer Profile' and 
                                customer_name.lower() in doc.get('customer_name', '').lower())]
                
                if customer_docs:
                    return f"Here's what I know about {customer_name}: {customer_docs[0]['content']}"
        
        # Default customer response
        if primary_doc.get('category') == 'Customer Profile':
            return f"Customer Information: {primary_doc['content']}"
        else:
            return f"Customer Data: {primary_doc['content'][:300]}..."
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with customer data only."""
        try:
            time.sleep(0.3)  # Simulate processing
            
            documents = self.retrieve_documents(query)
            
            if not documents:
                return {
                    'answer': "I couldn't find any customer information for your query. Try asking about TechCorp, HealthFirst, GreenEnergy, or RetailMax.",
                    'sources': [],
                    'query': query,
                    'status': 'no_results'
                }
            
            answer = self.generate_answer(query, documents)
            
            sources = []
            for doc in documents:
                sources.append({
                    'content': doc['content'],
                    'relevance_score': doc['relevance_score'],
                    'category': doc.get('category', 'Customer Data'),
                    'metadata': {
                        'title': doc['title'],
                        'category': doc.get('category', 'Customer Data'),
                        'id': doc['id']
                    }
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'query': query,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'query': query,
                'status': 'error'
            }
    
    def close(self):
        """Close the service."""
        if hasattr(self, 'mock_connector') and self.mock_connector:
            self.mock_connector.close()