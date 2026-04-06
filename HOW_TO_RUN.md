# NoAI Rephrase — How to Run

## Folder structure (put all files in the same folder)

```
noai-rephrase/
├── main.py                  ← updated backend (this file)
├── pipeline.py              ← your uploaded pipeline
├── requirements.txt         ← your uploaded requirements
├── noai-rephrase-v3.html    ← updated frontend (this file)
└── HOW_TO_RUN.md
```

---

## Step 1 — Get a free Gemini API key

1. Go to https://aistudio.google.com
2. Click **Get API key** → Create API key
3. Copy the key

---

## Step 2 — Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt

# Download spaCy model (one-time, ~12MB)
python -m spacy download en_core_web_sm
```

> ⚠️ First run also downloads sentence-transformers (~90MB) and GPT-2 (~500MB)
> automatically. This happens once and is then cached.

---

## Step 3 — Set your API key and start the backend

**Mac/Linux:**
```bash
export GEMINI_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_key_here"
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO │ noai │ Uvicorn running on http://127.0.0.1:8000
```

Verify it's working: open http://localhost:8000/health in your browser.
You should get: `{"status":"ok","service":"noai-rephrase"}`

---

## Step 4 — Open the frontend

Just open `noai-rephrase-v3.html` in your browser (double-click or drag into Chrome/Firefox).

> The HTML is configured to talk to `http://localhost:8000` automatically.

---

## How it works end-to-end

1. You paste AI text into the input box
2. After 1.2 seconds of no typing, it auto-calls `/api/score` and shows your AI % score live
3. Click **Humanize** → calls `/api/humanize` with your chosen mode
4. The pipeline runs: Extract anchors → Simplify → Rephrase → Validate (up to 3 retries)
5. Results appear: humanized text + before/after scores + diff view

---

## Endpoints

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/health` | GET | None | Healthcheck |
| `/api/score` | POST | None | Score text for AI-likeness |
| `/api/humanize` | POST | None | Full humanize pipeline |

### `/api/humanize` request body:
```json
{ "text": "your text here", "mode": "standard" }
```
Modes: `standard` · `aggressive` · `research`

### `/api/score` request body:
```json
{ "text": "your text here" }
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY not set` | Run `export GEMINI_API_KEY=...` before uvicorn |
| `ModuleNotFoundError: spacy` | Run `pip install -r requirements.txt` |
| `Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` |
| Frontend shows "Could not reach backend" | Make sure `uvicorn` is running on port 8000 |
| Slow on first request | GPT-2 / SBERT are loading into memory — wait ~30s |
| 429 error | Text exceeds 300-word limit — trim your input |
