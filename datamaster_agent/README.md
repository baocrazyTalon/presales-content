# Talon.One Presales Multi-Agent System

A LangGraph-powered multi-agent system that automates Technical Presales for Talon.One.  
It ingests raw notes and documents, then outputs high-fidelity, web-ready HTML/Tailwind presentations hosted on Vercel.

---

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                       LangGraph Graph                       │
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │ Coordinator │───►│  Data Architect  │───►│  Document │  │
│  │  (PM Agent) │◄───│  (RAG Agent)     │    │  Engineer │  │
│  └──────┬──────┘    └──────────────────┘    └─────┬─────┘  │
│         │            pgvector + Google Drive       │        │
│         │◄──────────────────────────────────────────        │
│         │  (validation loop)                                │
│         ▼                                                   │
│      Deploy ──► GitHub + Vercel                             │
└─────────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | Key Capabilities |
|---|---|---|
| **Coordinator** | Project Manager | Task decomposition, best-practices validation, routing |
| **Data Architect** | Knowledge Base | pgvector RAG, Google Drive (PDF/PPTX/XLSX), Vision LLM for images |
| **Document Engineer** | Full-Stack Dev | HTML/Tailwind generation, Mermaid.js diagrams, competitor style mimicry |

---

## Project Structure

```
presales_agent/
├── agents/
│   ├── coordinator.py          # Orchestrator & validator
│   ├── data_architect.py       # RAG retrieval agent
│   └── document_engineer.py   # HTML/Tailwind generator
├── core/
│   ├── state.py                # LangGraph AgentState (TypedDict)
│   ├── graph.py                # StateGraph wiring & routing logic
│   └── prompts.py              # System prompts for each agent
├── tools/
│   ├── google_drive.py         # Drive API: list, download, Vision LLM
│   ├── postgres_rag.py         # pgvector: ingest & similarity_search
│   └── deployment.py          # GitHub commit + Vercel trigger
├── docker-compose.yml          # PostgreSQL 16 + pgvector
├── init.sql                    # Schema: documents, presentations, agent_runs
├── requirements.txt
├── .env.example
└── main.py                     # CLI + FastAPI entry point
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI or Google API key
- GitHub personal access token
- Vercel project + token

### 2. Environment

```bash
cd presales_agent
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start the vector database

```bash
docker compose up -d
```

### 4. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. (Optional) Ingest existing documents

```python
from tools.postgres_rag import ingest_from_url

# Index Talon.One docs
ingest_from_url("https://docs.talon.one/docs/dev/integration-api/session-v2")
ingest_from_url("https://docs.talon.one/docs/product/account/managing-applications")
```

### 6. Run

```bash
# Interactive CLI
python main.py

# From a prospect JSON file
python main.py --prospect examples/dali.json

# As a REST API (POST /run)
python main.py --serve
```

---

## Prospect JSON format

```json
{
  "company_name": "Dali",
  "industry": "Retail",
  "use_case": "Unified loyalty program across POS and e-commerce",
  "competitor": "Salesforce Loyalty",
  "integrations": ["Braze", "Segment", "Talon.One API"],
  "raw_notes": "Prospect is currently using Salesforce Loyalty..."
}
```

---

## Talon.One Best Practices Checklist

The Coordinator validates every generated document against these criteria before deployment:

- MAC (Multi-Application Campaign) architecture explained
- Integration API Session V2 referenced
- Sub-50 ms latency claim included
- Unified rule engine vs. competitor "logic silos" addressed
- At least one Mermaid.js integration flow diagram
- CDP/CEP ecosystem coverage (Segment/mParticle · Braze/Iterable)
- Responsive HTML with Tailwind CSS

---

## REST API

Start with `python main.py --serve` then:

```http
POST http://localhost:8000/run
Content-Type: application/json

{
  "company_name": "My Muscle Chef",
  "industry": "Food & Beverage",
  "use_case": "Talon.One + loyalty cashback engine",
  "competitor": "Yotpo",
  "integrations": ["Talon.One", "Braze"],
  "raw_notes": ""
}
```

Response:
```json
{
  "vercel_url": "https://presales-abc123.vercel.app",
  "github_pr_url": "https://github.com/org/repo/blob/main/CLIENTS/GENERATED/my-muscle-chef-presales.html",
  "output_filename": "my-muscle-chef-presales.html",
  "validation_passed": true,
  "error": null
}
```

---

## Adding Knowledge to the Vector Store

```python
from tools.postgres_rag import ingest_document

ingest_document(
    content="Talon.One's Integration API Session V2 ...",
    source="talon_docs",
    source_id="integration-api-session-v2",
    title="Integration API — Session V2",
)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2 |
| LLM | GPT-4o / Gemini 1.5 Pro (multimodal) |
| Embeddings | text-embedding-3-small |
| Vector DB | PostgreSQL 16 + pgvector |
| Document parsing | PyPDF2, python-pptx, openpyxl, Pillow |
| Drive integration | Google Drive API v3 (OAuth 2.0) |
| Deployment | PyGithub + Vercel API |
| API server | FastAPI + Uvicorn |
