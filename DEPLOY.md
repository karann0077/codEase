# DevPilot — Step-by-Step Deployment Guide

Deploy backend on **Render** (free) and frontend on **Vercel** (free).
Total setup time: ~10 minutes.

---

## Step 1 — Push to GitHub

First, push this project to a GitHub repo (Render and Vercel both deploy from GitHub).

```bash
git init
git add .
git commit -m "Initial DevPilot commit"
git remote add origin https://github.com/YOUR_USERNAME/devpilot.git
git push -u origin main
```

---

## Step 2 — Deploy Backend on Render

### 2a. Create account
Go to [render.com](https://render.com) and sign up (free, no credit card).

### 2b. New Web Service
1. Dashboard → **New +** → **Web Service**
2. Connect your GitHub account → select your `devpilot` repo
3. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `devpilot-api` (or anything you like) |
| **Root Directory** | `backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Plan** | Free |

### 2c. Add Environment Variables
In the Render dashboard → your service → **Environment** tab, add:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | `gsk_...` your key from console.groq.com |
| `CHAT_MODEL` | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | `384` |
| `GITHUB_TOKEN` | optional — only for private repos |

### 2d. Deploy
Click **Create Web Service**. Wait ~3-5 minutes for first deploy.

### 2e. Get your backend URL
After deploy succeeds, you'll see a URL like:
```
https://devpilot-api.onrender.com
```
**Copy this URL — you need it for Step 3.**

Test it: open `https://devpilot-api.onrender.com/health` in your browser.
You should see: `{"status":"ok","groq_configured":true,...}`

> ⚠️ **Free tier note:** Render free services spin down after 15min of inactivity.
> First request after sleep takes ~30 seconds (cold start). This is normal.

---

## Step 3 — Deploy Frontend on Vercel

### 3a. Create account
Go to [vercel.com](https://vercel.com) and sign up with GitHub (free).

### 3b. New Project
1. Dashboard → **Add New** → **Project**
2. Import your `devpilot` GitHub repo
3. Configure:

| Setting | Value |
|---------|-------|
| **Root Directory** | `frontend` |
| **Framework Preset** | Vite |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |

### 3c. Add Environment Variable
Under **Environment Variables**, add:

| Key | Value |
|-----|-------|
| `VITE_API_URL` | `https://devpilot-api.onrender.com` ← your Render URL from Step 2e |

### 3d. Deploy
Click **Deploy**. Wait ~1-2 minutes.

Your app will be live at something like:
```
https://devpilot-xxxx.vercel.app
```

---

## Step 4 — Get Your Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. **API Keys** → **Create API Key**
4. Copy the key (`gsk_...`)
5. Paste into Render env var `GROQ_API_KEY`

**Groq free tier limits:**
- 14,400 requests/day
- 6,000 tokens/minute
- Multiple models available free

---

## Troubleshooting

### "API Offline" shown in the app sidebar
→ Your Render backend URL is wrong in Vercel env var, or the service hasn't started yet.
→ Check: open `YOUR_RENDER_URL/health` in browser — should return JSON.

### Render deploy fails with "ModuleNotFoundError"
→ Make sure **Root Directory** is set to `backend` in Render settings.

### Vercel build fails
→ Make sure **Root Directory** is set to `frontend` in Vercel settings.

### First request to backend is very slow
→ Normal on Render free tier — service cold starts after inactivity. Subsequent requests are fast.

### "GROQ_API_KEY not configured"
→ Add `GROQ_API_KEY` in Render dashboard → Environment → then redeploy.

### GitHub repo indexing times out
→ Large repos take time. Try a smaller repo first, or use file upload instead.
→ Render free tier has limited CPU — indexing ~50 files takes ~30s.

---

## Local Development (optional)

```bash
# Terminal 1 — Backend
cd backend
cp .env.example .env
# Edit .env: add your GROQ_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend  
cd frontend
# Create .env.local:
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm install
npm run dev
# Open http://localhost:3000
```

---

## Architecture

```
User Browser
    │
    ▼
Vercel (Frontend)
React + Vite + TypeScript
https://devpilot-xxx.vercel.app
    │
    │ HTTPS API calls
    ▼
Render (Backend)
FastAPI + Python
https://devpilot-api.onrender.com
    │
    ├── Groq API (LLM calls)
    │   └── llama-3.3-70b-versatile
    │
    ├── fastembed (local embeddings)
    │   └── BAAI/bge-small-en-v1.5
    │
    └── FAISS (in-memory vector search)
        └── per-session indexes
```
