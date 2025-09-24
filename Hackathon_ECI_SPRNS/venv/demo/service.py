"""
Demo Retrieval Service for SPRNS System.
Lightweight version that works without heavy ML dependencies.
"""

import time
import random
from typing import List, Dict, Any

class DemoRetrievalService:
    """
    Demo version of the retrieval service that simulates RAG functionality
    without requiring heavy ML dependencies.
    """
    
    def __init__(self):
        """Initialize the demo service."""
        self.knowledge_base = self._create_demo_knowledge_base()
        
        # Try to load mock graph data for more realistic responses
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from mock_graph.mock_neo4j import MockNeo4jConnector
            self.mock_connector = MockNeo4jConnector()
            self._enhance_knowledge_base_with_mock_data()
            print("Demo Retrieval Service initialized with mock customer data!")
        except Exception as e:
            self.mock_connector = None
            print(f"Demo Retrieval Service initialized (basic mode): {e}")
    
    def _enhance_knowledge_base_with_mock_data(self):
        """Enhance the knowledge base with data from the mock graph."""
        if not self.mock_connector:
            return
        
        # Add customer documents from mock graph
        for doc_id, doc in self.mock_connector.documents.items():
            enhanced_doc = {
                'id': doc['id'],
                'title': doc['title'],
                'content': doc['content'],
                'category': doc.get('type', 'General'),
                'keywords': doc.get('tags', []),
                'customer_id': doc.get('customer_id'),
                'created_date': doc.get('created_date')
            }
            self.knowledge_base.append(enhanced_doc)
        
        # Add customer information as searchable content
        for cust_id, customer in self.mock_connector.customers.items():
            # Get current address info
            current_address = self.mock_connector.get_current_address(cust_id)
            address_info = f"Currently located at {customer['current_location']}"
            if current_address:
                address_info += f" (moved {customer['address_changes']} times, last change: {customer['last_address_change']})"
            
            customer_doc = {
                'id': f"customer_{cust_id}",
                'title': f"Customer Profile: {customer['name']}",
                'content': f"{customer['name']} is a {customer['size']} {customer['industry']} company. {address_info}. Annual revenue: ${customer['annual_revenue']:,}. Status: {customer['status']}. Contact: {customer['contact_email']}. Last interaction: {customer['last_interaction']}. This customer has changed addresses {customer['address_changes']} times.",
                'category': 'Customer Profile',
                'keywords': [customer['industry'].lower(), customer['size'].lower(), 'customer', 'profile', 'address', 'location', customer['name'].lower().replace(' ', '_')],
                'customer_id': cust_id,
                'created_date': customer.get('created_date'),
                'customer_name': customer['name']  # Add this for easier matching
            }
            self.knowledge_base.append(customer_doc)
    
    def _create_demo_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create a minimal demo knowledge base focused on business/customer topics."""
        return [
            {
                'id': 'doc_business_1',
                'title': 'Customer Relationship Management',
                'content': 'Customer Relationship Management (CRM) involves managing interactions with current and potential customers. It includes tracking customer data, communication history, and business relationships to improve customer service and drive sales growth.',
                'category': 'Business',
                'keywords': ['CRM', 'customer', 'relationship', 'management', 'business']
            },
            {
                'id': 'doc_business_2',
                'title': 'Business Location Strategy',
                'content': 'Business location strategy involves selecting optimal locations for operations based on factors like cost, accessibility, talent availability, and market proximity. Companies often relocate to optimize operations, reduce costs, or access new markets.',
                'category': 'Business',
                'keywords': ['location', 'strategy', 'relocation', 'business', 'operations']
            },
            {
                'id': 'doc_business_3',
                'title': 'Enterprise Solutions Overview',
                'content': 'Enterprise solutions include analytics platforms, security suites, and IoT monitoring systems designed to help businesses optimize operations, ensure security, and gain insights from data. These solutions are tailored to specific industry needs.',
                'category': 'Business',
                'keywords': ['enterprise', 'solutions', 'analytics', 'security', 'IoT']
            }
        ]
    
    def _calculate_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate a smart relevance score between query and document."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check for customer-specific queries
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
        
        # Boost score for customer-specific documents
        customer_boost = 0
        for name, cust_id in customer_names.items():
            if name in query_lower and document.get('customer_id') == cust_id:
                customer_boost = 2.0  # Strong boost for exact customer match
                break
            elif name in query_lower and document.get('category') == 'Customer Profile':
                customer_boost = 1.5  # Medium boost for customer profiles
                break
        
        # Check title relevance
        title_words = set(document['title'].lower().split())
        title_overlap = len(query_words.intersection(title_words))
        
        # Check content relevance
        content_words = set(document['content'].lower().split())
        content_overlap = len(query_words.intersection(content_words))
        
        # Check keyword relevance
        keyword_overlap = 0
        if 'keywords' in document and document['keywords']:
            keyword_overlap = sum(1 for keyword in document['keywords'] 
                                if any(word in keyword.lower() for word in query_words))
        
        # Industry/category specific boosts
        industry_boost = 0
        if any(industry in query_lower for industry in ['technology', 'healthcare', 'energy', 'retail']):
            if document.get('category') == 'Customer Profile':
                industry_boost = 0.5
        
        # Special boost for industry overview queries
        if any(term in query_lower for term in ['industry', 'industries', 'serve', 'customers']):
            if document.get('category') == 'Customer Profile':
                industry_boost = 1.0
        
        # Product-specific boosts
        product_boost = 0
        product_terms = ['analytics', 'security', 'iot', 'platform', 'suite', 'system']
        if any(term in query_lower for term in product_terms):
            if any(term in document.get('content', '').lower() for term in product_terms):
                product_boost = 0.3
        
        # Address-specific boosts with special handling for specific location queries
        address_boost = 0
        address_terms = ['address', 'location', 'move', 'moved', 'relocation', 'relocate', 'change', 'history']
        
        # Check for specific location queries (these should prioritize customer profiles over history docs)
        specific_location_terms = ['second location', 'second address', 'first move', 'third location', 'original location', 'current location']
        is_specific_location_query = any(term in query_lower for term in specific_location_terms)
        
        if any(term in query_lower for term in address_terms):
            if is_specific_location_query and document.get('category') == 'Customer Profile':
                # Boost customer profiles for specific location queries
                address_boost = 2.5
            elif document.get('type') == 'address_history' and not is_specific_location_query:
                # Only boost history docs for general address queries
                address_boost = 1.5
            elif any(term in document.get('content', '').lower() for term in address_terms):
                address_boost = 0.5
        
        # Calculate base score
        base_score = (title_overlap * 0.4 + content_overlap * 0.4 + keyword_overlap * 0.2)
        
        # Apply boosts
        total_score = base_score + customer_boost + industry_boost + product_boost + address_boost
        
        # Heavily penalize generic documents when customer-specific query is detected
        if any(name in query_lower for name in customer_names.keys()):
            if document.get('category') not in ['Customer Profile', 'case_study', 'compliance_report', 'project_plan', 'roi_analysis', 'address_history']:
                total_score *= 0.1  # Heavily reduce relevance of generic documents
        
        # Penalize old generic categories completely for customer queries
        if any(name in query_lower for name in customer_names.keys()):
            if document.get('category') in ['AI/ML', 'Data Science', 'Programming', 'Database']:
                total_score = 0.01  # Almost eliminate these documents
        
        # Add small randomness for tie-breaking
        total_score += random.uniform(0, 0.05)
        
        return min(total_score, 3.0)  # Cap at 3.0 to allow for boosted scores
    
    def retrieve_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query with smart filtering."""
        query_lower = query.lower()
        
        # Calculate relevance scores
        scored_docs = []
        for doc in self.knowledge_base:
            relevance = self._calculate_relevance(query, doc)
            
            # Higher threshold for generic documents
            min_threshold = 0.1
            if doc.get('category') in ['AI/ML', 'Data Science', 'Programming', 'Database']:
                min_threshold = 1.5  # Much higher threshold for generic content
            
            if relevance > min_threshold:
                doc_with_score = doc.copy()
                doc_with_score['relevance_score'] = relevance
                scored_docs.append(doc_with_score)
        
        # Sort by relevance
        scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # For customer-specific queries, prioritize customer-related documents
        customer_names = ['techcorp', 'tech corp', 'healthfirst', 'health first', 
                         'greenenergy', 'green energy', 'retailmax', 'retail max']
        
        if any(name in query_lower for name in customer_names):
            # Separate customer-related and generic documents
            customer_docs = []
            business_docs = []
            generic_docs = []
            
            for doc in scored_docs:
                if (doc.get('category') == 'Customer Profile' or 
                    doc.get('customer_id') or
                    doc.get('type') in ['case_study', 'compliance_report', 'project_plan', 'roi_analysis', 'address_history']):
                    customer_docs.append(doc)
                elif doc.get('category') == 'Business':
                    business_docs.append(doc)
                else:
                    generic_docs.append(doc)
            
            # Return customer docs first, then business docs, avoid generic docs
            result = customer_docs[:limit]
            if len(result) < limit:
                result.extend(business_docs[:limit - len(result)])
            # Only add generic docs if absolutely necessary and with very low scores
            if len(result) < limit:
                filtered_generic = [doc for doc in generic_docs if doc.get('relevance_score', 0) > 1.0]
                result.extend(filtered_generic[:limit - len(result)])
            
            return result
        
        return scored_docs[:limit]
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate a smart answer based on retrieved documents and query type."""
        if not documents:
            return "I couldn't find any relevant information to answer your question. Try asking about our customers like TechCorp, HealthFirst, GreenEnergy, or RetailMax."
        
        query_lower = query.lower()
        
        # Check if this is a customer-specific query (expanded patterns)
        customer_queries = {
            'techcorp': 'TechCorp Solutions',
            'tech corp': 'TechCorp Solutions',
            'techcorp solutions': 'TechCorp Solutions',
            'healthfirst': 'HealthFirst Medical', 
            'health first': 'HealthFirst Medical',
            'healthfirst medical': 'HealthFirst Medical',
            'greenenergy': 'GreenEnergy Inc',
            'green energy': 'GreenEnergy Inc',
            'greenenergy inc': 'GreenEnergy Inc',
            'retailmax': 'RetailMax Chain',
            'retail max': 'RetailMax Chain',
            'retailmax chain': 'RetailMax Chain'
        }
        
        # PRIORITY: Check for location queries first (including natural language)
        location_indicators = ['location', 'address', 'where', 'move', 'moved', 'relocate', 'situated', 'based', 'headquarters']
        is_location_query = any(term in query_lower for term in location_indicators)
        
        specific_location_terms = ['second location', 'second address', 'first move', 'moved first', 
                                 'third location', 'third address', 'second move',
                                 'first location', 'original location', 'initial address',
                                 'current location', 'current address', 'where now']
        
        # Handle ANY location query (specific or general) for customers
        if is_location_query:
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
                            # Handle specific location requests
                            if any(term in query_lower for term in ['second location', 'second address', 'first move', 'moved first']):
                                if len(address_history) >= 2:
                                    second_location = address_history[1]
                                    return f"{customer_name} Second Location (First Move):\n\n📍 {second_location['address']}\n📅 Period: {second_location['start_date']} to {second_location['end_date']}\n💡 Reason: {second_location['reason']}\n\nThis was their first relocation from the original location."
                                else:
                                    return f"{customer_name} has not moved to a second location yet."
                            
                            elif any(term in query_lower for term in ['third location', 'third address', 'second move']):
                                if len(address_history) >= 3:
                                    third_location = address_history[2]
                                    return f"{customer_name} Third Location (Second Move):\n\n📍 {third_location['address']}\n📅 Period: {third_location['start_date']} to {third_location['end_date']}\n💡 Reason: {third_location['reason']}\n\nThis was their second relocation."
                                else:
                                    return f"{customer_name} has not moved to a third location yet."
                            
                            elif any(term in query_lower for term in ['first location', 'original location', 'initial address']):
                                first_location = address_history[0]
                                return f"{customer_name} Original Location:\n\n📍 {first_location['address']}\n📅 Period: {first_location['start_date']} to {first_location['end_date']}\n💡 Reason: {first_location['reason']}\n\nThis was their founding location."
                            
                            elif any(term in query_lower for term in ['current location', 'current address', 'where now']):
                                current_address = self.mock_connector.get_current_address(customer_id)
                                return f"{customer_name} Current Location:\n\n📍 {current_address['address']}\n📅 Since: {current_address['start_date']}\n💡 Reason: {current_address['reason']}\n\nThis is their current headquarters."
        
        # Find the most relevant document
        primary_doc = documents[0]
        
        # Generate customer-specific answers
        for customer_key, customer_name in customer_queries.items():
            if customer_key in query_lower:
                # Look for customer profile or customer-specific documents
                customer_docs = [doc for doc in documents if 
                               (doc.get('category') == 'Customer Profile' and 
                                customer_name.lower() in doc.get('customer_name', '').lower()) or
                               customer_name.lower() in doc.get('content', '').lower()]
                
                if customer_docs:
                    # Sort by relevance to get the best match
                    customer_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    customer_doc = customer_docs[0]
                    
                    if customer_doc.get('category') == 'Customer Profile':
                        return f"Here's what I know about {customer_name}: {customer_doc['content']}"
                    else:
                        return f"Regarding {customer_name}: {customer_doc['content'][:300]}..."
        
        # Check for product/solution queries
        if any(term in query_lower for term in ['analytics', 'platform']):
            analytics_docs = [doc for doc in documents if 'analytics' in doc.get('content', '').lower()]
            if analytics_docs:
                return f"Our Analytics Platform: {analytics_docs[0]['content'][:300]}..."
        
        if any(term in query_lower for term in ['security', 'suite']):
            security_docs = [doc for doc in documents if 'security' in doc.get('content', '').lower()]
            if security_docs:
                return f"Our Security Suite: {security_docs[0]['content'][:300]}..."
        
        if any(term in query_lower for term in ['iot', 'monitoring']):
            iot_docs = [doc for doc in documents if 'iot' in doc.get('content', '').lower()]
            if iot_docs:
                return f"Our IoT Monitoring System: {iot_docs[0]['content'][:300]}..."
        
        # Check for address/location queries
        if any(term in query_lower for term in ['address', 'location', 'move', 'moved', 'relocation', 'relocate', 'change', 'history']):
            # Check if it's a customer-specific address query
            for customer_key, customer_name in customer_queries.items():
                if customer_key in query_lower:
                    if hasattr(self, 'mock_connector') and self.mock_connector:
                        # Get address history from mock connector
                        customer_id = None
                        # Map customer names to IDs
                        name_to_id = {
                            'techcorp solutions': 'CUST_001',
                            'healthfirst medical': 'CUST_002', 
                            'greenenergy inc': 'CUST_003',
                            'retailmax chain': 'CUST_004'
                        }
                        customer_id = name_to_id.get(customer_name.lower())
                        
                        if customer_id:
                            address_history = self.mock_connector.get_customer_address_history(customer_id)
                            current_address = self.mock_connector.get_current_address(customer_id)
                            
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
                                    return f"{customer_name} Current Location:\n\n📍 {current_address['address']}\n📅 Since: {current_address['start_date']}\n💡 Reason: {current_address['reason']}\n\nThis is their current headquarters."
                                
                                else:
                                    # Default: Return full history
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
            
            # General address change analysis
            if hasattr(self, 'mock_connector') and self.mock_connector:
                if any(term in query_lower for term in ['all', 'customers', 'analysis', 'pattern']):
                    summary = self.mock_connector.get_analytics_summary()
                    return f"Address Change Analysis:\n\nTotal customers: {summary['total_customers']}\nTotal address changes: {summary['total_address_changes']}\nAverage changes per customer: {summary['avg_address_changes_per_customer']:.1f}\n\nAll customers have changed addresses 4 times each, showing high mobility in our customer base. Common reasons include business growth, cost optimization, compliance requirements, and strategic positioning."
        
        # Check for industry queries
        if 'industry' in query_lower or 'industries' in query_lower:
            # Use mock connector data if available
            if hasattr(self, 'mock_connector') and self.mock_connector:
                industries = set()
                customers_by_industry = {}
                
                for cust_id, customer in self.mock_connector.customers.items():
                    industry = customer.get('industry', 'Unknown')
                    industries.add(industry)
                    if industry not in customers_by_industry:
                        customers_by_industry[industry] = []
                    customers_by_industry[industry].append(customer['name'])
                
                industry_summary = []
                for industry in sorted(industries):
                    customers = customers_by_industry[industry]
                    industry_summary.append(f"{industry}: {', '.join(customers)}")
                
                return f"We serve customers across {len(industries)} industries:\n\n" + "\n".join(industry_summary)
            
            # Fallback to document-based approach
            customer_profiles = [doc for doc in documents if doc.get('category') == 'Customer Profile']
            if customer_profiles:
                industries = set()
                for doc in customer_profiles:
                    content = doc.get('content', '')
                    for industry in ['Technology', 'Healthcare', 'Energy', 'Retail']:
                        if industry in content:
                            industries.add(industry)
                
                if industries:
                    return f"We serve customers across multiple industries including: {', '.join(sorted(industries))}. Our customer base includes companies in technology, healthcare, energy, and retail sectors."
        
        # Default answer generation
        if primary_doc.get('category') == 'Customer Profile':
            return f"Customer Information: {primary_doc['content']}"
        elif primary_doc.get('type') in ['case_study', 'roi_analysis', 'compliance_report']:
            return f"Based on our customer experience: {primary_doc['content'][:300]}..."
        else:
            return f"According to our documentation: {primary_doc['content'][:300]}..."
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query end-to-end."""
        try:
            # Simulate processing time
            time.sleep(0.5)
            
            # Retrieve relevant documents
            documents = self.retrieve_documents(query)
            
            if not documents:
                return {
                    'answer': "I couldn't find any relevant documents to answer your question. Try rephrasing your query or asking about topics like machine learning, data science, or programming.",
                    'sources': [],
                    'query': query,
                    'status': 'no_results'
                }
            
            # Generate answer
            answer = self.generate_answer(query, documents)
            
            # Format sources for display
            sources = []
            for doc in documents:
                sources.append({
                    'content': doc['content'],
                    'relevance_score': doc['relevance_score'],
                    'category': doc.get('category', doc.get('type', 'Unknown')),
                    'metadata': {
                        'title': doc['title'],
                        'category': doc.get('category', doc.get('type', 'Unknown')),
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
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'sources': [],
                'query': query,
                'status': 'error'
            }
    
    def close(self):
        """Close the service and mock connector."""
        if hasattr(self, 'mock_connector') and self.mock_connector:
            self.mock_connector.close()