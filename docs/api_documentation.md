#  DocForge Hub API Documentation

Base URL

```
http://localhost:8000
```

All document APIs are under:

```
backend/api/routes/documents
```

---

# 📄 Documents API

## 1️⃣ Preview Document

Preview the document template structure from the registry.

**Endpoint**

```
POST /documents/preview
```

**Request Body**

```json
{
  "department": "engineering",
  "document_filename": "rfc"
}
```

**Response**

```json
{
  "document_name": "Request for Comments (RFC)",
  "department": "engineering",
  "sections": [
    {
      "name": "Summary",
      "mandatory": true
    }
  ]
}
```

---

## 2️⃣ Generate Document

Generate a new AI-powered document draft.

**Endpoint**

```
POST /documents/generate
```

**Request Body**

```json
{
  "department": "engineering",
  "document_filename": "rfc",
  "company_profile": {
    "company_name": "Acme Corp",
    "industry": "Technology"
  },
  "document_inputs": {
    "project_name": "AI Platform"
  }
}
```

**Response**

```json
{
  "draft_id": 1,
  "status": "draft_saved",
  "message": "Draft generated and stored successfully"
}
```

---

## 3️⃣ List Drafts

Retrieve all generated drafts.

**Endpoint**

```
GET /documents/drafts
```

**Response**

```json
[
  {
    "id": 1,
    "document_name": "Request for Comments (RFC)",
    "status": "NEEDS_REVIEW",
    "version": 1
  }
]
```

---

## 4️⃣ Delete Draft

Delete a draft from the system.

**Endpoint**

```
DELETE /documents/draft/{draft_id}
```

**Example**

```
DELETE /documents/draft/1
```

**Response**

```json
{
  "message": "Draft deleted successfully"
}
```

---

## 5️⃣ Get Draft Detail

Retrieve full draft including all sections.

**Endpoint**

```
GET /documents/draft/{draft_id}
```

**Example**

```
GET /documents/draft/1
```

**Response**

```json
{
  "id": 1,
  "document_name": "Request for Comments (RFC)",
  "status": "NEEDS_REVIEW",
  "version": 1,
  "sections": [
    {
      "section_name": "Summary",
      "blocks": [],
      "status": "draft"
    }
  ]
}
```

---

## 6️⃣ List Documents

List available documents for a department.

**Endpoint**

```
GET /documents/list
```

**Query Parameter**

```
department=engineering
```

**Example**

```
GET /documents/list?department=engineering
```

**Response**

```json
[
  {
    "document_name": "Request for Comments (RFC)",
    "internal_type": "policy"
  }
]
```

---

## 7️⃣ Export Draft

Export a document draft into a file format.

**Endpoint**

```
GET /documents/export/{draft_id}/{file_type}
```

**Example**

```
GET /documents/export/1/docx
```

**Supported Formats**

* docx (Microsoft Word)
* pdf (PDF Document)

 All sections must be **approved** before exporting.

---

## 8️⃣ Create Company Profile

Store company profile information.

**Endpoint**

```
POST /documents/company-profile
```

**Request**

```json
{
  "company_name": "Acme Corp",
  "industry": "Technology"
}
```

**Response**

```json
{
  "id": 1,
  "company_name": "Acme Corp",
  "industry": "Technology"
}
```

---

## 9️⃣ Approve Section

Approve a generated section before exporting.

**Endpoint**

```
POST /documents/approve-section
```

**Parameters**

```
draft_id
section_name
```

**Example**

```
draft_id=1
section_name=Summary
```

**Response**

```json
{
  "message": "Section approved successfully",
  "section_name": "Summary",
  "status": "approved"
}
```

---

## 🔟 Regenerate Section

Regenerate a section using AI with improvement instructions.

**Endpoint**

```
POST /documents/regenerate-section
```

**Parameters**

```
draft_id
section_name
improvement_note
```

**Example**

```
draft_id=1
section_name=Summary
improvement_note=Make the explanation clearer
```

**Response**

```json
{
  "message": "Section regenerated successfully"
}
```

---

## 1️⃣1️⃣ Save Section Edit

Save manual edits for a section.

**Endpoint**

```
POST /documents/save-section-edit
```

**Request Body**

```json
{
  "draft_id": 1,
  "section_name": "Summary",
  "updated_text": "Updated section content..."
}
```

**Response**

```json
{
  "message": "Section updated and improved successfully"
}
```

---

## 1️⃣2️⃣ Generate Clarification Questions

Generate AI-based clarification questions when required inputs are missing.

**Endpoint**

```
POST /documents/generate-questions
```

**Request Body**

```json
{
  "department": "engineering",
  "document_filename": "rfc",
  "company_profile": {},
  "document_inputs": {}
}
```

**Response**

```json
{
  "questions": [
    {
      "question": "What problem does this RFC address?",
      "field": "problem_statement"
    }
  ]
}
```

---

## 1️⃣3️⃣ Publish to Notion

Publish a finalized document to Notion.

**Endpoint**

```
POST /documents/publish-notion/{draft_id}
```

**Example**

```
POST /documents/publish-notion/1
```

**Response**

```json
{
  "message": "Published to Notion successfully"
}
```
---
## 1️⃣4️⃣ RAG Query

Ask questions using Retrieval-Augmented Generation.

**Endpoint**

```
POST /documents/rag-query
```
**Request Body**
```json
{
  "question": "What is the incident response process?",
  "session_id": "user_123",
  "doc_type": "policy",
  "industry": "technology"
}
```

**Response**
```json
{
  "answer": "The incident response process includes...",
  "sources": [
    {
      "document": "Incident Policy",
      "content": "..."
    }
  ]
}
```
---

## 1️⃣5️⃣ RAG Compare

Compare two documents using AI.

**Endpoint**

```
POST /documents/rag-compare

```
**Example**
```
POST /documents/rag-compare
```

**Request Body**
```json
{
  "doc_a": "Security Policy",
  "doc_b": "Incident Policy",
  "topic": "Access Control"
}
```

**Response**
```json
{
  "answer": "Comparison between documents...",
  "sources": []
}
```
---

## 1️⃣6️⃣ RAG Summarize

Generate a summary using RAG.

**Endpoint**
```
POST /documents/rag-summarize
```
**Example**
```
POST /documents/rag-summarize
```
**Request Body**
```json
{
  "query": "Summarize compliance policies",
  "doc_type": "policy",
  "industry": "finance"
}
```
**Response**
```json
{
  "summary": "The compliance policies include..."
}
```
---

## 1️⃣7️⃣ RAG Evaluate

Evaluate RAG system performance.

**Endpoint**
```
POST/documents/rag-evaluate
```
**Example**
```
POST /documents/rag-evaluate
```
**Response**
```json
{
  "message": "Evaluation completed",
  "data": [
    {
      "faithfulness": 0.92,
      "answer_relevancy": 0.88
    }
  ],
  "avg_faithfulness": 0.91,
  "avg_relevancy": 0.87,
  "avg_context_precision": 0.89,
  "avg_context_recall": 0.85
}
```
---

## 1️⃣8️⃣ StateCase Assistant (Conversational AI)

This endpoint powers the StateCase AI Assistant.
It supports:

* Conversational queries
* Memory & follow-ups
* Clarification questions
* Ticket creation
* Out-of-domain detection
* Cached responses

---

### **Endpoint**

```
POST /documents/statecase-chat
```

---

### **Request Body**

```json
{
  "question": "What are the working hours for remote employees?",
  "session_id": "user_123",
  "doc_type": "policy",
  "industry": "technology",
  "version": "latest"
}
```

---

### **Successful Response (Answer Found)**

```json
{
  "answer": "Remote employees must follow core working hours from 10:00 AM to 4:00 PM.",
  "confidence": 63,
  "sources": [
    "Remote Work Policy → Work Hours & Availability"
  ],
  "ticket_created": false,
  "cache_hit": false
}
```

---

### **Out-of-Domain Response**

```json
{
  "answer": "This question is outside the scope of available documents.",
  "confidence": 0,
  "sources": [],
  "ticket_created": false
}
```

---

### **Low Confidence → Ticket Created**

```json
{
  "answer": "I couldn't find a reliable answer. A support ticket has been created.",
  "confidence": 45,
  "ticket_created": true,
  "ticket_id": "a1b2c3d4"
}
```

---

### **Duplicate Ticket Detected**

```json
{
  "answer": "A ticket for this query already exists. No duplicate created.",
  "confidence": 50,
  "ticket_created": false
}
```

---

### **Clarification Required**

```json
{
  "answer": "Could you provide more details? Which policy or department are you referring to?",
  "confidence": 0
}
```

---

### **Meta (Chat History)**

```json
{
  "answer": "Your previous question was: 'Explain remote work policy'",
  "confidence": 100
}
```

---

### **Features Covered by This Endpoint**

* Multi-turn memory
* RAG-based retrieval
* Semantic duplicate ticket detection
* Out-of-domain filtering
* Intelligent clarification
* Redis caching

---

## 🎫 Ticket Management APIs (StateCase)

These endpoints manage the lifecycle of support tickets created by the StateCase assistant.

---

### 1️⃣ Update Ticket Status

Update the status of an existing ticket (e.g., Open → In Progress → Closed).

---

#### **Endpoint**

```id="s4g5v2"
POST /documents/update-ticket
```

---

#### **Request Body**

```json id="s7d2k9"
{
  "ticket_id": "a1b2c3d4",
  "status": "closed"
}
```

---

#### **Response**

```json id="k9d3w1"
{
  "success": true,
  "message": "Updated to Closed"
}
```

---

#### **Notes**

* `ticket_id` must exist in the system
* `status` can be:

  * `open`
  * `in_progress`
  * `closed`

---

### 2️⃣ Check Ticket Status

Retrieve the current status of a ticket.

---

#### **Endpoint**

```id="d3j9x1"
GET /documents/ticket-status/{ticket_id}
```

---

#### **Example**

```id="x8m2p7"
GET /documents/ticket-status/a1b2c3d4
```

---

#### **Response**

```json id="l2q8z5"
{
  "ticket_id": "a1b2c3d4",
  "status": "in_progress"
}
```

---

#### **Fallback Response**

```json id="f8n1c4"
{
  "ticket_id": "a1b2c3d4",
  "status": "unknown"
}
```

---

## 🔄 Ticket Lifecycle

```id="y6k2t9"
User Query → Low Confidence
        ↓
Ticket Created
        ↓
Update Status (via API)
        ↓
Resolved / Closed
```

---

## 🧠 Integration with StateCase

* Tickets are automatically created by:

  * `/documents/statecase-chat`
* These APIs allow:

  * Manual updates by admin
  * Integration with dashboards
  * Tracking ticket resolution status


# 📘 Interactive API Docs

Swagger UI

```
http://localhost:8000/docs
```


