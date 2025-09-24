# SPRNS Customer Test Guide

## 🎯 Your SPRNS UI is now running with Mock Customer Data!

### 🌐 Access the Dashboard
**URL:** http://localhost:8501

### 📊 Available Test Data

#### 4 Mock Customers:

1. **TechCorp Solutions**
   - Industry: Technology
   - Size: Large
   - Current Location: San Francisco, CA (moved 4 times)
   - Revenue: $50M
   - Status: Active
   - Products: Enterprise Analytics Platform
   - Address History: Palo Alto → Mountain View → San Jose → Fremont → San Francisco

2. **HealthFirst Medical**
   - Industry: Healthcare  
   - Size: Medium
   - Current Location: Boston, MA (moved 4 times)
   - Revenue: $15M
   - Status: Active
   - Products: Cloud Security Suite
   - Address History: Cambridge → Somerville → Brookline → Newton → Boston

3. **GreenEnergy Inc**
   - Industry: Energy
   - Size: Small
   - Current Location: Austin, TX (moved 4 times)
   - Revenue: $5M
   - Status: Prospect
   - Products: IoT Monitoring System (planned)
   - Address History: Round Rock → Cedar Park → Pflugerville → Georgetown → Austin

4. **RetailMax Chain**
   - Industry: Retail
   - Size: Large
   - Current Location: Chicago, IL (moved 4 times)
   - Revenue: $75M
   - Status: Active
   - Products: Enterprise Analytics Platform
   - Address History: Schaumburg → Naperville → Aurora → Joliet → Chicago

### 🔍 Test Queries to Try

#### Customer Information Queries:
- "Tell me about TechCorp Solutions"
- "What is HealthFirst Medical?"
- "Show me GreenEnergy Inc details"
- "RetailMax Chain information"
- "What industries do we serve?"

#### Product & Implementation Queries:
- "Show me analytics success stories"
- "How did HealthFirst implement security?"
- "What are the benefits of our IoT system?"
- "Analytics ROI at RetailMax"
- "Security compliance at HealthFirst"

#### Business Intelligence Queries:
- "Which customers use analytics?"
- "Show me healthcare customers"
- "What products do we offer?"
- "Customer success stories"
- "Implementation case studies"

#### Address Change Queries (NEW!):
- "TechCorp address history" (full history)
- "Where is HealthFirst located?" (current location)
- "GreenEnergy location changes" (full history)
- "RetailMax address moves" (full history)
- "Show me all customer address changes" (analysis)
- "Address change analysis" (patterns)
- "Why did customers move locations?" (reasons)

#### Specific Location Queries (NEW!):
- "HealthFirst second location" (first move)
- "TechCorp second address" (first move)
- "GreenEnergy first move" (second location)
- "RetailMax third location" (second move)
- "TechCorp original location" (founding address)
- "HealthFirst current location" (current address only)

### ✅ Expected Results

When you ask about customers, you should now see:
- **Accurate customer information** (not generic definitions)
- **Relevant sources** showing customer profiles and case studies
- **High relevance scores** (2.0-3.0) for customer-specific content
- **Proper source attribution** with customer names and document types

### 🚀 Features Working:
- ✅ Customer-specific query handling
- ✅ Smart relevance scoring
- ✅ Industry analysis
- ✅ Product implementation stories
- ✅ Source attribution with relevance scores
- ✅ Mock graph database integration
- ✅ **Address change tracking (4 moves per customer)**
- ✅ **Location history with dates and reasons**
- ✅ **Address change analytics and patterns**

### 🛑 To Stop the Server:
```bash
# Find and kill the process
ps aux | grep streamlit
kill [process_id]
```

Or press Ctrl+C in the terminal where you started it.

---

**🎉 Your SPRNS system is now ready for customer data testing!**