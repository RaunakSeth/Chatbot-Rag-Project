# Business AI Chatbot Template

A reusable, multi-tenant RAG chatbot template for small/medium businesses.
Each client gets an isolated instance scoped to their own content.
**No model is ever fine-tuned or retrained per client.**

---

## Architecture

```
User message
     │
     ▼
┌─────────────────────────────┐
│ Stage 1: Scope Classifier    │──┐  (parallel)
└─────────────────────────────┘  │
┌─────────────────────────────┐  │
│ Stage 2: Safety Guardrail    │──┘
└─────────────────────────────┘
     │
     ▼  merge results
Out of scope OR unsafe → return refusal, STOP
     │
     └── in-scope & safe
          ▼
┌─────────────────────────────┐
│ Stage 3: Retrieval           │  BGE-M3 embed → LanceDB
└─────────────────────────────┘
          ▼
┌─────────────────────────────┐
│ Stage 4: Generation          │  RAG answer via Ollama/llama.cpp
└─────────────────────────────┘
          ▼
┌─────────────────────────────┐
│ Stage 5: Humanization        │  prompt-level only (no extra model)
└─────────────────────────────┘
          ▼
     Response to user
```

---

## Hardware Tiers

| Tier | Hardware | Generation model |
|---|---|---|
| **A (default)** | CPU-only, 8–16 GB RAM | `Qwen3.5-4B-Instruct` (Q4_K_M GGUF) |
| **B** | Single small GPU, 8–16 GB VRAM | `Qwen3-8B-Instruct` (Q4_K_M GGUF) |

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set HARDWARE_TIER and OLLAMA_BASE_URL
```

### 3. Pull generation model via Ollama

```bash
# Tier A
ollama pull qwen2.5:4b-instruct-q4_K_M
# Tier B
ollama pull qwen2.5:8b-instruct-q4_K_M
```

### 4. Onboard a client

```bash
python -m onboarding.onboard \
  --client-id acme_corp \
  --business-name "Acme Corp" \
  --url https://acmecorp.com \
  --pdf-dir ./docs/acme \
  --tone friendly
```

This will:
1. Crawl the website
2. Parse all PDFs in `--pdf-dir`
3. Chunk, embed (BGE-M3), and index into `clients/acme_corp/lancedb/`
4. Write `clients/acme_corp/config.yaml`

### 5. Start the API server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Send a chat message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "acme_corp",
    "session_id": "user-session-123",
    "message": "What are your pricing plans?"
  }'
```

---

## Client Config (`clients/<id>/config.yaml`)

```yaml
client_id: acme_corp
business_name: "Acme Corp"
hardware_tier: A        # A or B
tone: friendly          # friendly | formal | concise
refusal_message: "I can only answer questions about Acme Corp."
prohibited_topics:
  - politics
  - competitors
retrieval:
  top_k: 5
  score_threshold: 0.35
session:
  max_history_turns: 6
```

---

## Project Layout

```
chatbot/
├── config.py          # Config schema + loader
├── pipeline.py        # 5-stage orchestrator
└── stages/
    ├── stage1_classifier.py   # Qwen3-0.6B-Instruct
    ├── stage2_safety.py       # Qwen3-Guard-0.6B / 4B
    ├── stage3_retrieval.py    # BGE-M3 + LanceDB
    ├── stage4_generation.py   # Qwen3.5-4B / 8B via Ollama
    └── stage5_humanizer.py    # Prompt-only humanization
onboarding/
├── crawler.py         # Website → text chunks
├── pdf_parser.py      # PDF → text chunks
└── onboard.py         # CLI entry point
api/
├── app.py             # FastAPI application
└── routes.py          # /chat, /health, /sessions
```

---

## Explicit Non-Goals (per spec)

- ❌ No fine-tuning or LoRA per client
- ❌ No merging Stage 1 + Stage 2 into one call
- ❌ No shared multi-tenant GPU server
- ❌ No closed-API as permanent generation layer
