#  DevPilot - AI Developer Productivity Suite

> **Documentation · Debugging · Code Intelligence**
> Index any GitHub repo and get instant AI-powered docs, debugging, and code chat.

---

## 🚀 What is DevPilot?

DevPilot is a full stack AI powered developer tool that solves three problems developers face daily:

- 📄 **Code is hard to understand** → Auto-generate README style docs, flowcharts, and commented code
- 🐛 **Bugs take too long to find** → AI root cause analysis with fixed code in seconds
- 💬 **Documentation never gets written** → RAG based chat grounded in your actual codebase

Built with **Groq LLM** (free tier), **FAISS vector search**, and **fastembed** — runs 100% free.

---

## ✨ Features

### 📄 1. Documentation Helper

Index any GitHub repo and click any file to generate:

| Tab | What you get |
|-----|-------------|
| **Overview** | README style summary, architecture explanation, API docs, key concepts, potential issues, complexity |
| **Functions** | Every function documented with purpose, parameters, returns, step-by-step logic, full docstring |
| **Commented Code** | Full source with inline comments on every function, class, loop, and condition |
| **Flowchart** | Live-rendered Mermaid.js control flow diagram |

**Two modes:**
- **Repo Files** — GitHub style folder tree, click any file to document it
- **Custom Code** — Paste any snippet into the Monaco editor

---

### 🐛 2. AI Debugger

Four debugging modes:

#### 🔍 Analyze Error
Paste stacktrace + error message + optional code context and console logs.

**Returns:** error type · root cause · explanation · exact fix · fixed code snippet · prevention advice · confidence level

#### 🖼️ Multimodal Debug
Upload a **screenshot** + stacktrace + console logs. AI correlates all three signals to find the exact file and line.

**Returns:** screenshot interpretation · most likely file:line · root cause · step-by-step fix

#### 🔎 Decode Minified Stacktrace
Paste a minified JS stacktrace (e.g. `at t.e (main.8f3a2.js:1:4521)`). Optionally add source map.

**Returns:** decoded frames · original file:line references · confidence per frame · plain-English summary

#### 💻 Custom Code Debug
Monaco editor  paste your buggy code + error message. AI sees your actual source and gives precise root cause + fix.

---

### 💬 3. Code Chat (RAG)

Chat with your codebase. Every answer is grounded in your actual code — not hallucinated from training data.

**How it works:**
1. Index a GitHub repo or upload files
2. Your question → embedded to 384 dim vector via `fastembed`
3. FAISS searches indexed chunks → top 3 semantically similar chunks returned
4. Chunks injected into Groq prompt with filename + similarity score
5. AI answers with full awareness of your real code

**Example questions:**
```
"How does authentication work in this project?"
"Where is the database connection set up?"
"Find all places where API calls are made"
"What does the UserService class do?"
"What are the main API endpoints?"
```

Mermaid flowcharts render live inside chat responses.

---

## 🏗️ Architecture
```
Browser (Vercel)
     │
     │  HTTPS — axios (30s normal / 300s ingestion timeout)
     ▼
FastAPI Backend (Render, Python 3.11.9)
     │
     ├── Groq API ──────── LLM inference (llama-3.3-70b-versatile)
     │                     Vision (llama-3.2-11b-vision-preview)
     │
     ├── fastembed ──────── Local ONNX embeddings (BAAI/bge-small-en-v1.5, 384-dim)
     │                      Cached at /tmp/fastembed_cache
     │
     └── FAISS ─────────── IndexFlatIP, in-memory, per-session
                           L2 normalized for cosine similarity
```

---

## 🛠️ Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| LLM | `llama-3.3-70b-versatile` | Groq API, free tier |
| Vision | `llama-3.2-11b-vision-preview` | Groq vision model for screenshots |
| Embeddings | `BAAI/bge-small-en-v1.5` | fastembed local ONNX, 384-dim |
| Vector DB | FAISS `IndexFlatIP` | In-memory, per-session, cosine similarity |
| Backend | FastAPI + Python 3.11.9 | Async, deployed on Render free tier |
| Frontend | React + TypeScript + Vite | Monaco editor, Mermaid.js, Vercel |
| RAG Pipeline | CodeRAG-faithful | Chunk-level indexing, concurrent embeddings |

---

## 📁 Project Structure
```
codEase/
├── backend/
│   ├── main.py           # FastAPI app — all endpoints, lifespan startup, CORS
│   ├── ai_engine.py      # All Groq calls via run_in_executor (async-safe)
│   ├── embeddings.py     # fastembed wrapper — sync + async embedding functions
│   ├── ingestion.py      # GitHub Trees API ingestion + code-aware chunking
│   ├── vector_index.py   # FAISS per-session index — add, search, stats
│   ├── config.py         # pydantic-settings config from env vars
│   ├── requirements.txt
│   ├── render.yaml
│   └── runtime.txt       # python-3.11.9
└── frontend/
    ├── src/
    │   ├── App.tsx        # All pages: Docs, Debugger, Chat
    │   ├── lib/api.ts     # Axios clients — 30s normal / 300s ingestion
    │   ├── types/index.ts # TypeScript types
    │   └── index.css      # Full dark theme
    ├── vercel.json
    └── vite.config.ts
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API status + groq_configured bool |
| `POST` | `/session/create` | Create new UUID session |
| `DELETE` | `/session/{id}` | Clear FAISS index for session |
| `GET` | `/session/{id}/stats` | List indexed files |
| `POST` | `/ingest/github` | Index a public GitHub repo |
| `POST` | `/ingest/files` | Index uploaded files |
| `GET` | `/fetch-file` | Fetch raw file from GitHub for docs |
| `POST` | `/docs/explain` | Generate documentation JSON |
| `POST` | `/docs/commented-code` | Generate commented source (lazy load) |
| `POST` | `/debug/analyze` | Analyze error + stacktrace |
| `POST` | `/debug/multimodal` | Correlate screenshot + logs + stack |
| `POST` | `/debug/decode-stacktrace` | Decode minified JS stacktrace |
| `POST` | `/chat` | RAG chat with indexed codebase |

---

## ⚙️ RAG Pipeline (CodeRAG-Faithful)

### Ingestion
1. Parse `owner/repo` from GitHub URL
2. `GET api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1` — **one API call** for full file tree
3. Filter with `should_index()` — skips `node_modules`, `__pycache__`, `.git`, lock files, files >50KB
4. Cap at **40 files** (Render free tier RAM limit ~512MB)
5. **Concurrent download** via `asyncio.Semaphore(10)` + `asyncio.gather`
6. **Code-aware chunking:**
   - Python → splits on `def` / `class` / `async def` boundaries
   - JS/TS → splits on `function` / `class` / `export` boundaries
   - Others → splits on blank lines
7. Each chunk embedded concurrently via `asyncio.gather`
8. All chunks added to `FAISS IndexFlatIP` with L2 normalization

### Query
1. User question → `fastembed` → 384-dim vector → L2 normalized
2. FAISS inner product search → top 3 chunks
3. Context format: `File / Path / Similarity: 0.847 / Content`
4. Groq LLM generates grounded answer



## 🧪 Demo Repo — TaskFlow

A small Python task manager CLI with **9 real bugs** — built to demo all three DevPilot features.

**Repo:** [`github.com/karann0077/demopr`](https://github.com/karann0077/demopr)

| # | File | Bug | Error Type |
|---|------|-----|-----------|
| 1 | `task_manager.py` | `deadline` stored as `str` not `datetime` | TypeError |
| 2 | `task_manager.py` | `is_overdue()` compares `datetime > str` | TypeError |
| 3 | `task_manager.py` | `get_task()` missing KeyError handling | KeyError |
| 4 | `auth.py` | `validate_password()` never returns `True` | Logic Bug |
| 5 | `auth.py` | MD5 used for password hashing | Security Bug |
| 6 | `auth.py` | Session timeout: minutes compared to seconds | Logic Bug |
| 7 | `stats.py` | `calculate_average([])` → division by zero | ZeroDivisionError |
| 8 | `stats.py` | Wrong median for even-length lists | Logic Bug |
| 9 | `stats.py` | Priority distribution returns decimals not % | Logic Bug |

**Demo flow:**
1. **Code Chat** → index repo → ask *"What bugs exist in this codebase?"*
2. **Debugger → Analyze Error** → paste the stacktrace below
3. **Debugger → Custom Code** → paste `task_manager.py` → get root cause + fixed code
4. **Documentation** → click `stats.py` → see docs, flowchart, commented code

**Demo stacktrace:**
```
Traceback (most recent call last):
  File "main.py", line 44, in main
    overdue = manager.get_overdue_tasks()
  File "src/task_manager.py", line 52, in get_overdue_tasks
    return [t for t in self.tasks.values() if not t.completed and t.is_overdue()]
  File "src/task_manager.py", line 28, in is_overdue
    return datetime.now() > self.deadline
TypeError: '>' not supported between instances of 'datetime.datetime' and 'str'
```

---

## ⚠️ Known Limitations

| Limitation | Details |
|-----------|---------|
| No persistence | FAISS index lives in RAM — Render restarts (every ~15min idle) wipe all indexed data |
| Public repos only | `raw.githubusercontent.com` requires public repos — private needs GitHub token |
| 40 file cap | Large repos auto-capped to protect free tier RAM (~512MB) |
| 50KB file limit | Files over 50KB skipped during ingestion |
| No auth | No user accounts — anyone with the URL can use the deployment |
| Groq rate limits | Free tier per-minute limits may affect heavy concurrent use |


<div align="center">
  <strong>Built with ❤️ by <a href="https://github.com/karann0077">karann0077</a></strong><br/>
  <sub>Groq · FAISS · fastembed · FastAPI · React · TypeScript · Vercel · Render</sub>
</div>
