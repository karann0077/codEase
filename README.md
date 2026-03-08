# DevPilot — AI Developer Productivity Suite

> Documentation Helper · AI Debugger · RAG-Powered Code Chat

**100% free to run** — uses Groq's free LLM API + local sentence-transformers embeddings (no OpenAI needed).

---

## Tech Stack

| Layer | Tech | Cost |
|-------|------|------|
| LLM (chat/debug/docs) | Groq API — llama-3.3-70b | Free tier |
| Vision (screenshots) | Groq — llama-3.2-11b-vision | Free tier |
| Embeddings | sentence-transformers (local CPU) | Free / no API |
| Vector search | FAISS in-memory | Free |
| Backend | FastAPI + Python | Free on Render |
| Frontend | React + Vite + TypeScript | Free on Vercel |
| Editor | Monaco Editor | Free |
| Diagrams | Mermaid.js | Free |

---

## Get Your Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Dashboard → API Keys → Create API Key
4. Copy the key starting with `gsk_...`

**Free tier limits** (as of 2025):
- 14,400 requests/day
- 6,000 tokens/minute per model
- Multiple models available

---

## Local Development

### 1. Backend

```bash
cd backend
cp .env.example .env
# Edit .env — paste your GROQ_API_KEY

pip install -r requirements.txt

# First run downloads the embedding model (~90MB, one time only)
uvicorn main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### 2. Frontend

```bash
cd frontend
cp .env.example .env.local
# Set: VITE_API_URL=http://localhost:8000

npm install
npm run dev
# Runs at http://localhost:3000
```

---

## Deploy to Render (Backend)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo, set **Root Directory** = `backend`
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. Environment Variables:
   | Key | Value |
   |-----|-------|
   | `GROQ_API_KEY` | `gsk_...` your Groq key |
   | `CHAT_MODEL` | `llama-3.3-70b-versatile` |
   | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
   | `EMBEDDING_DIM` | `384` |
   | `GITHUB_TOKEN` | optional, for private repos |

> ⚠️ **Note on Render Free:** First request after inactivity takes ~30s to spin up (cold start). sentence-transformers model (~90MB) downloads on first build.

---

## Deploy to Vercel (Frontend)

1. Go to [vercel.com](https://vercel.com) → New Project → Import your repo
2. Set **Root Directory** = `frontend`
3. Add Environment Variable:
   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | `https://devpilot-api.onrender.com` (your Render URL) |
4. Deploy

---

## Environment Variables

### Backend (`backend/.env`)
```env
GROQ_API_KEY=gsk_your_key_here

# Model options (all free on Groq):
# llama-3.3-70b-versatile  ← recommended, best quality
# llama-3.1-8b-instant     ← fastest responses
# mixtral-8x7b-32768       ← large context window
# gemma2-9b-it             ← Google Gemma 2
CHAT_MODEL=llama-3.3-70b-versatile

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384

GITHUB_TOKEN=ghp_optional_for_private_repos
```

### Frontend (`frontend/.env.local`)
```env
VITE_API_URL=http://localhost:8000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health + Groq status |
| POST | `/session/create` | New session |
| POST | `/ingest/github` | Index GitHub repo |
| POST | `/ingest/files` | Index file list |
| POST | `/ingest/upload` | Multipart upload |
| POST | `/docs/explain` | Explain + document code |
| POST | `/debug/analyze` | Error analysis + fix |
| POST | `/debug/multimodal` | Screenshot + logs + stacktrace |
| POST | `/debug/decode-stacktrace` | Decode minified JS/TS |
| POST | `/chat` | RAG chat with codebase |

---

## Credits

- **ChatDBG** (Emery Berger et al.) — structured AI debugging prompts
- **CodeRAG** — FAISS vector search + RAG pipeline
- **Groq** — blazing fast free LLM inference
- **sentence-transformers** — free local embeddings
