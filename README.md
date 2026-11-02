# AI Output Quality Evaluation Dashboard (Python)

Streamlit + SQLite + Plotly app that scores AI responses on Relevance, Completeness, Clarity and Factual Support (0-10 each).
Overall = average x 10 (0-100): 90+ Excellent, 75-89 Good, 60-74 Average, below 60 Needs Improvement.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. The SQLite file `evaluations.db` is created on first run with 30 labelled sample records.

## Pages
Overview (KPIs, trend, status mix, recent-evaluation selector) · New Evaluation (validated form, edit support) ·
Evaluation History (search, filters, click a row to open it; edit, delete, CSV and PDF export) ·
Analytics (7/30/90 days or all time) · Compare Responses (new prompt + two responses, or two saved evaluations) ·
Reports (CSV and detailed PDF) · Settings.

## Scoring modes
- **Demo mode (default):** deterministic text heuristic (keyword overlap, length, sentence structure). It is NOT an AI judge.
  Factual Support stays low and is flagged "unverified" unless sources or an expected answer are given; nothing is ever marked verified.
- **AI judge (optional):** Supports Google Gemini (`GEMINI_API_KEY`), Anthropic Claude (`ANTHROPIC_API_KEY`), or FreeLLMAPI / OpenAI-compatible endpoints (`FREELLMAPI_API_KEY` / `FREELLMAPI_BASE_URL`). Either copy `.env.example` to `.env` and paste your key there, or paste it on the Settings page. Never commit `.env`. If the call fails, the app falls back to demo mode and records that.

## Files
`app.py` UI · `core.py` scoring, feedback, SQLite, CSV/PDF · `requirements.txt`

Environment variables: `EVAL_DB` (database path), `EVAL_NO_SEED=1` (start empty).
