"""
Mock Neo4j Graph Database for SPRNS System.
Simulates Neo4j behavior with 4 customers and related data for local testing.
"""

import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class MockNeo4jConnector:
    """
    Mock Neo4j connector that simulates graph database operations
    with a predefined dataset of 4 customers and related information.
    """
    
    def __init__(self, uri: str = None, username: str = None, password: str = None, database: str = None):
        """Initialize the mock connector with sample data."""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.connected = True
        
        # Initialize mock data
        self._init_mock_data()
        print("Mock Neo4j Connector initialized with sample customer data")
    
    def _init_mock_data(self):
        """Initialize the mock graph database with customers and related data."""
        
        # Customer nodes with address history
        self.customers = {
            "CUST_001": {
                "id": "CUST_001",
                "name": "TechCorp Solutions",
                "industry": "Technology",
                "size": "Large",
                "current_location": "San Francisco, CA",
                "annual_revenue": 50000000,
                "contact_email": "contact@techcorp.com",
                "phone": "+1-555-0101",
                "status": "Active",
                "created_date": "2022-01-15",
                "last_interaction": "2024-09-20",
                "address_changes": 4,
                "last_address_change": "2024-06-15"
            },
            "CUST_002": {
                "id": "CUST_002",
                "name": "HealthFirst Medical",
                "industry": "Healthcare",
                "size": "Medium",
                "current_location": "Boston, MA",
                "annual_revenue": 15000000,
                "contact_email": "info@healthfirst.com",
                "phone": "+1-555-0202",
                "status": "Active",
                "created_date": "2022-03-22",
                "last_interaction": "2024-09-18",
                "address_changes": 4,
                "last_address_change": "2024-03-10"
            },
            "CUST_003": {
                "id": "CUST_003",
                "name": "GreenEnergy Inc",
                "industry": "Energy",
                "size": "Small",
                "current_location": "Austin, TX",
                "annual_revenue": 5000000,
                "contact_email": "hello@greenenergy.com",
                "phone": "+1-555-0303",
                "status": "Prospect",
                "created_date": "2023-06-10",
                "last_interaction": "2024-09-15",
                "address_changes": 4,
                "last_address_change": "2024-08-01"
            },
            "CUST_004": {
                "id": "CUST_004",
                "name": "RetailMax Chain",
                "industry": "Retail",
                "size": "Large",
                "current_location": "Chicago, IL",
                "annual_revenue": 75000000,
                "contact_email": "business@retailmax.com",
                "phone": "+1-555-0404",
                "status": "Active",
                "created_date": "2021-11-08",
                "last_interaction": "2024-09-22",
                "address_changes": 4,
                "last_address_change": "2024-01-20"
            }
        }
        
        # Address history for each customer (4 address changes each)
        self.address_history = {
            "CUST_001": [  # TechCorp Solutions
                {
                    "address_id": "ADDR_001_1",
                    "address": "123 Startup Ave, Palo Alto, CA 94301",
                    "address_type": "Headquarters",
                    "start_date": "2022-01-15",
                    "end_date": "2022-08-30",
                    "reason": "Initial office setup",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_001_2", 
                    "address": "456 Innovation Blvd, Mountain View, CA 94041",
                    "address_type": "Headquarters",
                    "start_date": "2022-09-01",
                    "end_date": "2023-05-15",
                    "reason": "Expansion - needed larger office space",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_001_3",
                    "address": "789 Tech Park Dr, San Jose, CA 95110", 
                    "address_type": "Headquarters",
                    "start_date": "2023-05-16",
                    "end_date": "2024-01-10",
                    "reason": "Cost optimization - moved to cheaper location",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_001_4",
                    "address": "321 Enterprise Way, Fremont, CA 94538",
                    "address_type": "Headquarters", 
                    "start_date": "2024-01-11",
                    "end_date": "2024-06-14",
                    "reason": "Temporary relocation during office renovation",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_001_5",
                    "address": "555 Market St, San Francisco, CA 94105",
                    "address_type": "Headquarters",
                    "start_date": "2024-06-15", 
                    "end_date": None,
                    "reason": "Permanent move to premium downtown location",
                    "is_current": True
                }
            ],
            "CUST_002": [  # HealthFirst Medical
                {
                    "address_id": "ADDR_002_1",
                    "address": "100 Medical Center Dr, Cambridge, MA 02139",
                    "address_type": "Main Clinic",
                    "start_date": "2022-03-22",
                    "end_date": "2022-11-30",
                    "reason": "Initial clinic location",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_002_2",
                    "address": "250 Healthcare Plaza, Somerville, MA 02143",
                    "address_type": "Main Clinic", 
                    "start_date": "2022-12-01",
                    "end_date": "2023-07-20",
                    "reason": "Moved for better patient accessibility",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_002_3",
                    "address": "75 Wellness Blvd, Brookline, MA 02446",
                    "address_type": "Main Clinic",
                    "start_date": "2023-07-21", 
                    "end_date": "2023-12-15",
                    "reason": "Compliance with new healthcare regulations",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_002_4",
                    "address": "180 Research Park, Newton, MA 02458",
                    "address_type": "Main Clinic",
                    "start_date": "2023-12-16",
                    "end_date": "2024-03-09",
                    "reason": "Partnership with research facility",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_002_5",
                    "address": "500 Commonwealth Ave, Boston, MA 02215",
                    "address_type": "Main Clinic",
                    "start_date": "2024-03-10",
                    "end_date": None,
                    "reason": "Consolidated operations in central Boston location",
                    "is_current": True
                }
            ],
            "CUST_003": [  # GreenEnergy Inc
                {
                    "address_id": "ADDR_003_1", 
                    "address": "50 Solar Way, Round Rock, TX 78664",
                    "address_type": "Operations Center",
                    "start_date": "2023-06-10",
                    "end_date": "2023-09-30",
                    "reason": "Initial startup location",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_003_2",
                    "address": "125 Wind Farm Rd, Cedar Park, TX 78613", 
                    "address_type": "Operations Center",
                    "start_date": "2023-10-01",
                    "end_date": "2024-02-15",
                    "reason": "Moved closer to wind farm projects",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_003_3",
                    "address": "300 Green Tech Blvd, Pflugerville, TX 78660",
                    "address_type": "Operations Center", 
                    "start_date": "2024-02-16",
                    "end_date": "2024-05-30",
                    "reason": "Shared facility with other green tech companies",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_003_4",
                    "address": "88 Renewable Dr, Georgetown, TX 78626",
                    "address_type": "Operations Center",
                    "start_date": "2024-05-31",
                    "end_date": "2024-07-31", 
                    "reason": "Temporary location during facility construction",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_003_5",
                    "address": "777 Energy Plaza, Austin, TX 78701",
                    "address_type": "Headquarters",
                    "start_date": "2024-08-01",
                    "end_date": None,
                    "reason": "Established permanent headquarters in Austin downtown",
                    "is_current": True
                }
            ],
            "CUST_004": [  # RetailMax Chain
                {
                    "address_id": "ADDR_004_1",
                    "address": "1000 Retail Row, Schaumburg, IL 60173",
                    "address_type": "Corporate HQ",
                    "start_date": "2021-11-08", 
                    "end_date": "2022-06-30",
                    "reason": "Original headquarters location",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_004_2",
                    "address": "2500 Commerce Dr, Naperville, IL 60563",
                    "address_type": "Corporate HQ",
                    "start_date": "2022-07-01",
                    "end_date": "2023-03-15",
                    "reason": "Moved for tax incentives and larger warehouse space",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_004_3", 
                    "address": "450 Distribution Blvd, Aurora, IL 60502",
                    "address_type": "Corporate HQ",
                    "start_date": "2023-03-16",
                    "end_date": "2023-10-31",
                    "reason": "Consolidated with distribution center",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_004_4",
                    "address": "750 Logistics Lane, Joliet, IL 60431",
                    "address_type": "Corporate HQ", 
                    "start_date": "2023-11-01",
                    "end_date": "2024-01-19",
                    "reason": "Strategic move closer to transportation hubs",
                    "is_current": False
                },
                {
                    "address_id": "ADDR_004_5",
                    "address": "1200 Michigan Ave, Chicago, IL 60605",
                    "address_type": "Corporate HQ",
                    "start_date": "2024-01-20",
                    "end_date": None,
                    "reason": "Premium downtown location for brand prestige",
                    "is_current": True
                }
            ]
        }
        
        # Product nodes
        self.products = {
            "PROD_001": {
                "id": "PROD_001",
                "name": "Enterprise Analytics Platform",
                "category": "Software",
                "price": 50000,
                "description": "Advanced analytics platform for enterprise data processing and insights"
            },
            "PROD_002": {
                "id": "PROD_002",
                "name": "Cloud Security Suite",
                "category": "Security",
                "price": 25000,
                "description": "Comprehensive cloud security solution with threat detection and prevention"
            },
            "PROD_003": {
                "id": "PROD_003",
                "name": "IoT Monitoring System",
                "category": "IoT",
                "price": 15000,
                "description": "Real-time monitoring system for IoT devices and sensor networks"
            }
        }
        
        # Document nodes (knowledge base)
        self.documents = {
            "DOC_001": {
                "id": "DOC_001",
                "title": "TechCorp Implementation Guide",
                "content": "TechCorp Solutions successfully implemented our Enterprise Analytics Platform in Q2 2023. The implementation involved migrating 500TB of historical data and training 200+ users. Key success factors included dedicated project management, phased rollout approach, and comprehensive user training. The platform now processes over 1M transactions daily and has reduced reporting time by 75%.",
                "type": "case_study",
                "customer_id": "CUST_001",
                "created_date": "2023-07-15",
                "tags": ["implementation", "analytics", "success_story"]
            },
            "DOC_002": {
                "id": "DOC_002",
                "title": "HealthFirst Security Compliance Report",
                "content": "HealthFirst Medical achieved HIPAA compliance using our Cloud Security Suite. The implementation included data encryption, access controls, audit logging, and staff training. All security requirements were met within 90 days. The system now monitors 50+ healthcare applications and has prevented 15 security incidents in the first year.",
                "type": "compliance_report",
                "customer_id": "CUST_002",
                "created_date": "2023-05-20",
                "tags": ["security", "compliance", "healthcare", "HIPAA"]
            },
            "DOC_003": {
                "id": "DOC_003",
                "title": "GreenEnergy IoT Deployment Plan",
                "content": "GreenEnergy Inc is planning to deploy our IoT Monitoring System across 25 wind farms. The project will monitor 500+ turbines, track energy production, and predict maintenance needs. Expected benefits include 20% reduction in downtime and 15% increase in energy efficiency. Implementation timeline is 6 months with pilot testing in Q1 2024.",
                "type": "project_plan",
                "customer_id": "CUST_003",
                "created_date": "2023-12-01",
                "tags": ["IoT", "energy", "monitoring", "wind_farms"]
            },
            "DOC_004": {
                "id": "DOC_004",
                "title": "RetailMax Analytics ROI Analysis",
                "content": "RetailMax Chain's use of our Enterprise Analytics Platform has delivered significant ROI. Key metrics: 25% improvement in inventory turnover, 30% reduction in stockouts, $2M annual savings in operational costs. The platform analyzes data from 150 stores and processes 10M customer transactions monthly. Customer satisfaction scores increased by 18%.",
                "type": "roi_analysis",
                "customer_id": "CUST_004",
                "created_date": "2024-01-30",
                "tags": ["analytics", "retail", "ROI", "inventory"]
            },
            "DOC_005": {
                "id": "DOC_005",
                "title": "Product Comparison: Analytics vs Security",
                "content": "Our Enterprise Analytics Platform and Cloud Security Suite serve different but complementary needs. Analytics Platform focuses on data insights, reporting, and business intelligence with features like real-time dashboards, predictive analytics, and custom reporting. Security Suite provides threat detection, compliance monitoring, and access control. Many customers benefit from both solutions working together.",
                "type": "product_guide",
                "customer_id": None,
                "created_date": "2024-02-15",
                "tags": ["product_comparison", "analytics", "security"]
            },
            "DOC_006": {
                "id": "DOC_006",
                "title": "Industry Trends: Healthcare Technology",
                "content": "Healthcare industry is rapidly adopting digital solutions. Key trends include telemedicine growth (300% increase), AI-powered diagnostics, IoT medical devices, and enhanced cybersecurity measures. Compliance requirements like HIPAA drive security investments. Our solutions help healthcare organizations modernize while maintaining compliance and patient data protection.",
                "type": "industry_report",
                "customer_id": None,
                "created_date": "2024-03-10",
                "tags": ["healthcare", "trends", "digital_transformation", "compliance"]
            },
            "DOC_007": {
                "id": "DOC_007",
                "title": "TechCorp Address Change History",
                "content": "TechCorp Solutions has undergone 4 address changes since inception. Started in Palo Alto (2022), moved to Mountain View for expansion (2022), relocated to San Jose for cost optimization (2023), temporarily moved to Fremont during renovation (2024), and finally established permanent headquarters in downtown San Francisco (2024). Each move was strategically planned to support business growth and operational efficiency.",
                "type": "address_history",
                "customer_id": "CUST_001",
                "created_date": "2024-06-20",
                "tags": ["address_change", "relocation", "business_growth", "techcorp"]
            },
            "DOC_008": {
                "id": "DOC_008", 
                "title": "HealthFirst Medical Facility Relocations",
                "content": "HealthFirst Medical has changed locations 4 times to better serve patients and comply with regulations. Initial clinic in Cambridge (2022), moved to Somerville for accessibility (2022), relocated to Brookline for compliance (2023), partnered with Newton research facility (2023), and consolidated operations in central Boston (2024). Each move improved patient care and operational efficiency.",
                "type": "address_history",
                "customer_id": "CUST_002", 
                "created_date": "2024-03-15",
                "tags": ["address_change", "healthcare", "patient_care", "compliance"]
            },
            "DOC_009": {
                "id": "DOC_009",
                "title": "GreenEnergy Inc Location Evolution",
                "content": "GreenEnergy Inc has moved 4 times since startup. Started in Round Rock (2023), moved to Cedar Park near wind farms (2023), shared facility in Pflugerville (2024), temporary location in Georgetown during construction (2024), and established permanent Austin headquarters (2024). Each relocation supported their renewable energy mission and business expansion.",
                "type": "address_history", 
                "customer_id": "CUST_003",
                "created_date": "2024-08-05",
                "tags": ["address_change", "startup_growth", "renewable_energy", "austin"]
            },
            "DOC_010": {
                "id": "DOC_010",
                "title": "RetailMax Chain Corporate Moves",
                "content": "RetailMax Chain has relocated headquarters 4 times for strategic advantages. Original HQ in Schaumburg (2021), moved to Naperville for tax benefits (2022), consolidated with Aurora distribution center (2023), relocated to Joliet near transportation hubs (2023), and established premium Chicago downtown location (2024). Each move optimized operations and enhanced brand presence.",
                "type": "address_history",
                "customer_id": "CUST_004",
                "created_date": "2024-01-25", 
                "tags": ["address_change", "retail", "logistics", "brand_positioning"]
            },
            "DOC_011": {
                "id": "DOC_011",
                "title": "Customer Address Change Impact Analysis",
                "content": "Analysis of customer address changes shows common patterns: 1) Startups move frequently for growth (avg 4 moves), 2) Healthcare moves for compliance and accessibility, 3) Energy companies relocate near resources, 4) Retail chains optimize for logistics and brand. Address changes impact service delivery, requiring updated contact information, revised contracts, and adjusted support protocols.",
                "type": "analysis_report",
                "customer_id": None,
                "created_date": "2024-09-01",
                "tags": ["address_change", "customer_analysis", "business_impact", "patterns"]
            }
        }
        
        # Relationships between entities
        self.relationships = [
            # Customer-Product relationships (purchases/interests)
            {"from": "CUST_001", "to": "PROD_001", "type": "PURCHASED", "date": "2023-02-15", "amount": 50000},
            {"from": "CUST_002", "to": "PROD_002", "type": "PURCHASED", "date": "2023-04-10", "amount": 25000},
            {"from": "CUST_003", "to": "PROD_003", "type": "INTERESTED_IN", "date": "2023-11-20", "probability": 0.75},
            {"from": "CUST_004", "to": "PROD_001", "type": "PURCHASED", "date": "2022-08-05", "amount": 50000},
            {"from": "CUST_001", "to": "PROD_002", "type": "INTERESTED_IN", "date": "2024-01-15", "probability": 0.60},
            
            # Customer-Document relationships
            {"from": "CUST_001", "to": "DOC_001", "type": "SUBJECT_OF", "relevance": 1.0},
            {"from": "CUST_002", "to": "DOC_002", "type": "SUBJECT_OF", "relevance": 1.0},
            {"from": "CUST_003", "to": "DOC_003", "type": "SUBJECT_OF", "relevance": 1.0},
            {"from": "CUST_004", "to": "DOC_004", "type": "SUBJECT_OF", "relevance": 1.0},
            
            # Product-Document relationships
            {"from": "PROD_001", "to": "DOC_001", "type": "FEATURED_IN", "relevance": 0.9},
            {"from": "PROD_001", "to": "DOC_004", "type": "FEATURED_IN", "relevance": 0.9},
            {"from": "PROD_002", "to": "DOC_002", "type": "FEATURED_IN", "relevance": 0.9},
            {"from": "PROD_003", "to": "DOC_003", "type": "FEATURED_IN", "relevance": 0.9},
            
            # Industry relationships
            {"from": "CUST_002", "to": "DOC_006", "type": "RELEVANT_TO", "relevance": 0.8},
        ]
    
    def close(self):
        """Close the mock connection."""
        self.connected = False
    
    def verify_connectivity(self):
        """Verify mock connection (always succeeds)."""
        if not self.connected:
            raise Exception("Mock connection closed")
        return True
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a mock Cypher query and return results.
        Simulates common Neo4j query patterns.
        """
        parameters = parameters or {}
        query = query.strip()
        
        # Simple query parsing for common patterns
        if "MATCH (d:Document)" in query and "RETURN d" in query:
            return self._handle_document_query(query, parameters)
        elif "MATCH (c:Customer)" in query:
            return self._handle_customer_query(query, parameters)
        elif "MATCH (p:Product)" in query:
            return self._handle_product_query(query, parameters)
        elif "MERGE" in query:
            return self._handle_merge_query(query, parameters)
        else:
            # Default: return empty result
            return []
    
    def _handle_document_query(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle document-related queries."""
        results = []
        
        # Check for content search
        if "d.content CONTAINS" in query and "query_text" in parameters:
            search_term = parameters["query_text"].lower()
            limit = parameters.get("limit", 10)
            
            for doc_id, doc in self.documents.items():
                if (search_term in doc["content"].lower() or 
                    search_term in doc["title"].lower() or
                    any(search_term in tag.lower() for tag in doc["tags"])):
                    results.append({"d": doc})
            
            return results[:limit]
        
        # Return all documents if no specific filter
        limit = parameters.get("limit", len(self.documents))
        for doc_id, doc in list(self.documents.items())[:limit]:
            results.append({"d": doc})
        
        return results
    
    def _handle_customer_query(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle customer-related queries."""
        results = []
        
        # Check for specific customer ID
        if "c.id =" in query and "customer_id" in parameters:
            customer_id = parameters["customer_id"]
            if customer_id in self.customers:
                results.append({"c": self.customers[customer_id]})
        else:
            # Return all customers
            for customer_id, customer in self.customers.items():
                results.append({"c": customer})
        
        return results
    
    def _handle_product_query(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle product-related queries."""
        results = []
        
        for product_id, product in self.products.items():
            results.append({"p": product})
        
        return results
    
    def _handle_merge_query(self, query: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle MERGE operations (simulate adding data)."""
        # For mock purposes, just return success
        return [{"status": "success"}]
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """Add a document to the mock graph."""
        doc_id = document.get("id")
        if not doc_id:
            raise ValueError("Document must have an 'id' field")
        
        self.documents[doc_id] = document
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        return self.documents.get(doc_id)
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a customer by their ID."""
        return self.customers.get(customer_id)
    
    def get_related_documents(self, doc_id: str, relationship_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents related to the given document."""
        related_docs = []
        
        # Find relationships involving this document
        for rel in self.relationships:
            if rel["from"] == doc_id or rel["to"] == doc_id:
                if relationship_type is None or rel["type"] == relationship_type:
                    # Get the other entity in the relationship
                    other_id = rel["to"] if rel["from"] == doc_id else rel["from"]
                    
                    # If it's a document, add it to results
                    if other_id in self.documents:
                        related_docs.append(self.documents[other_id])
        
        return related_docs[:limit]
    
    def search_documents(self, properties: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents with matching properties."""
        results = []
        
        for doc_id, doc in self.documents.items():
            match = True
            for key, value in properties.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            
            if match:
                results.append(doc)
        
        return results[:limit]
    
    def get_customer_documents(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all documents related to a specific customer."""
        customer_docs = []
        
        # Direct customer documents
        for doc_id, doc in self.documents.items():
            if doc.get("customer_id") == customer_id:
                customer_docs.append(doc)
        
        # Documents related through relationships
        for rel in self.relationships:
            if rel["from"] == customer_id and rel["to"] in self.documents:
                customer_docs.append(self.documents[rel["to"]])
        
        return customer_docs
    
    def get_customer_products(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all products associated with a customer."""
        customer_products = []
        
        for rel in self.relationships:
            if (rel["from"] == customer_id and 
                rel["to"] in self.products and 
                rel["type"] in ["PURCHASED", "INTERESTED_IN"]):
                
                product = self.products[rel["to"]].copy()
                product["relationship"] = rel["type"]
                product["relationship_data"] = {k: v for k, v in rel.items() if k not in ["from", "to", "type"]}
                customer_products.append(product)
        
        return customer_products
    
    def search_customers_by_industry(self, industry: str) -> List[Dict[str, Any]]:
        """Search customers by industry."""
        results = []
        
        for customer_id, customer in self.customers.items():
            if customer.get("industry", "").lower() == industry.lower():
                results.append(customer)
        
        return results
    
    def get_customer_address_history(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get the complete address history for a customer."""
        return self.address_history.get(customer_id, [])
    
    def get_current_address(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get the current address for a customer."""
        history = self.address_history.get(customer_id, [])
        for address in history:
            if address.get("is_current", False):
                return address
        return None
    
    def get_address_changes_count(self, customer_id: str) -> int:
        """Get the number of address changes for a customer."""
        customer = self.customers.get(customer_id)
        return customer.get("address_changes", 0) if customer else 0
    
    def search_customers_by_address_changes(self, min_changes: int = 0) -> List[Dict[str, Any]]:
        """Search customers by number of address changes."""
        results = []
        for customer_id, customer in self.customers.items():
            if customer.get("address_changes", 0) >= min_changes:
                customer_with_history = customer.copy()
                customer_with_history["address_history"] = self.address_history.get(customer_id, [])
                results.append(customer_with_history)
        return results
    
    def get_specific_location(self, customer_id: str, location_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific location by number (1=first/original, 2=second, etc.)."""
        history = self.address_history.get(customer_id, [])
        if 1 <= location_number <= len(history):
            return history[location_number - 1]  # Convert to 0-based index
        return None
    
    def get_second_location(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get the second location (first move) for a customer."""
        return self.get_specific_location(customer_id, 2)
    
    def get_first_location(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get the original/first location for a customer."""
        return self.get_specific_location(customer_id, 1)
    
    def get_third_location(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get the third location (second move) for a customer."""
        return self.get_specific_location(customer_id, 3)
    
    def get_address_change_timeline(self) -> List[Dict[str, Any]]:
        """Get a timeline of all address changes across all customers."""
        timeline = []
        for customer_id, history in self.address_history.items():
            customer_name = self.customers[customer_id]["name"]
            for address in history:
                timeline.append({
                    "customer_id": customer_id,
                    "customer_name": customer_name,
                    "address": address["address"],
                    "start_date": address["start_date"],
                    "end_date": address["end_date"],
                    "reason": address["reason"],
                    "is_current": address["is_current"]
                })
        
        # Sort by start date
        timeline.sort(key=lambda x: x["start_date"])
        return timeline
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get a summary of the mock data for analytics."""
        total_address_changes = sum(c.get("address_changes", 0) for c in self.customers.values())
        
        return {
            "total_customers": len(self.customers),
            "total_products": len(self.products),
            "total_documents": len(self.documents),
            "total_relationships": len(self.relationships),
            "total_address_changes": total_address_changes,
            "avg_address_changes_per_customer": total_address_changes / len(self.customers),
            "customers_by_industry": {
                industry: len([c for c in self.customers.values() if c.get("industry") == industry])
                for industry in set(c.get("industry") for c in self.customers.values())
            },
            "customers_by_status": {
                status: len([c for c in self.customers.values() if c.get("status") == status])
                for status in set(c.get("status") for c in self.customers.values())
            },
            "customers_by_address_changes": {
                str(changes): len([c for c in self.customers.values() if c.get("address_changes") == changes])
                for changes in set(c.get("address_changes", 0) for c in self.customers.values())
            }
        }