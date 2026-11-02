"""Scoring, feedback, SQLite storage and PDF/CSV export for the evaluation dashboard."""
import io, json, os, random, re, sqlite3
from datetime import datetime, timedelta

try:  # optional: read keys from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k not in os.environ:
                        os.environ[k] = v

DB_PATH = os.environ.get("EVAL_DB", "evaluations.db")
DIMS = ["relevance", "completeness", "clarity", "factual"]
LABELS = {"relevance": "Relevance", "completeness": "Completeness",
          "clarity": "Clarity", "factual": "Factual Support"}
BANDS = [("Excellent", 90, "#16a34a"), ("Good", 75, "#2563eb"),
         ("Average", 60, "#d97706"), ("Needs Improvement", 0, "#dc2626")]
STATUS_COLORS = {n: c for n, _, c in BANDS}


def clamp(x):
    return max(0, min(10, int(x + 0.5)))


def overall(scores):
    """Overall score as a percentage: average of four 0-10 scores x 10."""
    return int(sum(scores[k] for k in DIMS) / 4 * 10 + 0.5)


def status(p):
    return next(n for n, lo, _ in BANDS if p >= lo)


# ---------------------------------------------------------------- scoring
def _tok(s):
    return set(re.findall(r"[a-z0-9]{4,}", s.lower()))


def _overlap(a, b):
    A, B = _tok(a), _tok(b)
    return len(A & B) / len(A) if A else 0.0


def score_demo(prompt, response, expected="", sources=""):
    """Deterministic heuristic. NOT an AI judgement and it never verifies facts."""
    words = len(response.split())
    sents = [s for s in re.split(r"[.!?]+\s", response) if s.strip()]
    avg_len = words / max(1, len(sents))
    base = min(1, words / 120)
    sc = {"relevance": clamp(2 + _overlap(prompt, response) * 11)}
    sc["completeness"] = clamp(2 + ((_overlap(expected, response) * .6 + base * .4) if expected else base) * 8)
    structure = 1 if re.search(r"\n|^\s*[-*\d]", response, re.M) else 0
    sc["clarity"] = clamp(9 - (max(0, (avg_len - 28) / 3)) - (2 if avg_len < 6 else 0)
                          - (2 if words < 25 else 0) + structure)
    n_src = len(re.findall(r"https?://|\n", sources)) + (1 if sources.strip() else 0)
    if sources.strip():
        sc["factual"] = clamp(5 + min(3, n_src) + _overlap(sources, response) * 2)
    elif expected:
        sc["factual"] = clamp(3 + _overlap(expected, response) * 4)
    else:
        sc["factual"] = 3
    return sc


def feedback(scores, has_sources):
    why = {
        "relevance": ["The response barely addresses the query.", "Partly on topic but drifts from the question.",
                      "Mostly aligned with the question.", "Directly addresses the query and stays on topic."],
        "completeness": ["Major parts of the request are missing.", "Several requested details are absent.",
                         "Most points covered; some detail missing.", "Covers all key aspects of the request."],
        "clarity": ["Hard to follow; structure is weak.", "Readable but sentences or structure need work.",
                    "Well organised and easy to read.", "Very clear, concise and well structured."],
        "factual": ["Claims are unverified and unsupported by references.", "Limited support; several claims need sources.",
                    "Mostly plausible; some claims need references.", "Claims align with the supplied reference material."],
    }
    out = {"why": {}, "strengths": [], "weaknesses": [], "recs": []}
    for k in DIMS:
        v = scores[k]
        out["why"][k] = why[k][0 if v < 4 else 1 if v < 6 else 2 if v < 8 else 3]
        line = f"{LABELS[k]} ({v}/10): {out['why'][k]}"
        if v >= 8:
            out["strengths"].append(line)
        elif v < 6:
            out["weaknesses"].append(line)
    if scores["factual"] < 8: out["recs"].append("Add supporting references and verify key claims.")
    if scores["completeness"] < 8: out["recs"].append("Cover the missing details from the request.")
    if scores["clarity"] < 8: out["recs"].append("Use shorter sentences and a clearer structure.")
    if scores["relevance"] < 8: out["recs"].append("Tighten the answer to the question asked.")
    out["strengths"] = out["strengths"] or ["No dimension scored 8 or above."]
    out["weaknesses"] = out["weaknesses"] or ["No dimension scored below 6."]
    out["recs"] = out["recs"] or ["Keep this standard; consider adding citations for extra assurance."]
    if not has_sources:
        out["why"]["factual"] = ("Unverified: no reference sources were supplied, so factual accuracy "
                                 "could not be checked. " + out["why"]["factual"])
    return out


def active_ai_provider():
    explicit = os.environ.get("AI_PROVIDER", "").strip().lower()
    if explicit in ("freellmapi", "gemini", "anthropic", "openai"):
        return "freellmapi" if explicit == "openai" else explicit
    if os.environ.get("FREELLMAPI_API_KEY"):
        return "freellmapi"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "freellmapi"
    return None


def ai_available():
    return active_ai_provider() is not None


def _call_gemini(system, user):
    import urllib.request, urllib.error
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(body)
            msg = err_json.get("error", {}).get("message", body)
        except Exception:
            msg = body
        raise RuntimeError(f"Gemini API ({e.code}): {msg}") from e


def _call_openai_compatible(system, user):
    import urllib.request, urllib.error
    api_key = os.environ.get("FREELLMAPI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("FREELLMAPI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "http://localhost:3001/v1")
    model = os.environ.get("FREELLMAPI_MODEL") or os.environ.get("OPENAI_MODEL", "auto")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FreeLLMAPI ({e.code}): {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"FreeLLMAPI gateway unreachable at {base_url} ({e.reason})") from e


def _call_anthropic(system, user):
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("EVAL_MODEL", "claude-sonnet-5-5"), max_tokens=1200,
        system=system, messages=[{"role": "user", "content": user}])
    return msg.content[0].text


def score_with_ai(prompt, response, expected="", sources=""):
    """Evaluates response using Gemini, Anthropic, or FreeLLMAPI/OpenAI."""
    provider = active_ai_provider()
    system = ("You are a strict evaluator of AI responses. Return ONLY JSON: "
              '{"scores":{"relevance":0-10,"completeness":0-10,"clarity":0-10,"factual":0-10},'
              '"why":{"relevance":"","completeness":"","clarity":"","factual":""},'
              '"strengths":[],"weaknesses":[],"recs":[]}. Integers only. If no reference sources or expected '
              "answer are provided, factual must be 5 or lower and its explanation must say it is unverified. "
              "Never claim facts are verified without the supplied material.")
    user = (f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n\nEXPECTED ANSWER:\n{expected or 'none'}"
            f"\n\nREFERENCE SOURCES:\n{sources or 'none'}")

    if provider == "gemini":
        raw_text = _call_gemini(system, user)
    elif provider == "freellmapi":
        raw_text = _call_openai_compatible(system, user)
    elif provider == "anthropic":
        raw_text = _call_anthropic(system, user)
    else:
        raise ValueError("No AI provider configured.")

    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        raise ValueError(f"Could not parse JSON from AI response: {raw_text[:200]}")
    data = json.loads(match.group(0))
    sc = {k: clamp(float(data["scores"][k])) for k in DIMS}
    if not (sources.strip() or expected):
        sc["factual"] = min(sc["factual"], 5)
    fb = {k: data.get(k, []) for k in ("strengths", "weaknesses", "recs")}
    fb["why"] = {k: str(data["why"][k]) for k in DIMS}
    return sc, fb


def evaluate(prompt, response, expected="", sources="", use_ai=False):
    """Returns (scores, feedback, mode). Falls back to demo mode if the AI call fails."""
    if use_ai and ai_available():
        provider = active_ai_provider()
        try:
            sc, fb = score_with_ai(prompt, response, expected, sources)
            return sc, fb, f"ai ({provider})"
        except Exception as exc:  # noqa: BLE001
            sc = score_demo(prompt, response, expected, sources)
            err_msg = str(exc)
            if len(err_msg) > 70:
                err_msg = err_msg[:67] + "..."
            return sc, feedback(sc, bool(sources.strip())), f"demo (AI failed: {err_msg})"
    sc = score_demo(prompt, response, expected, sources)
    return sc, feedback(sc, bool(sources.strip())), "demo"


# ---------------------------------------------------------------- storage
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS evaluations(
            id TEXT PRIMARY KEY, name TEXT, created_at TEXT, prompt TEXT, response TEXT,
            expected TEXT, sources TEXT, scores TEXT, feedback TEXT, mode TEXT)""")
        if c.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0 and not os.environ.get("EVAL_NO_SEED"):
            seed_samples(c)


def _row(r):
    d = dict(r)
    d["scores"], d["feedback"] = json.loads(d["scores"]), json.loads(d["feedback"])
    d["created_at"] = datetime.fromisoformat(d["created_at"])
    d["overall"] = overall(d["scores"])
    d["status"] = status(d["overall"])
    return d


def all_evals():
    with _conn() as c:
        return [_row(r) for r in c.execute("SELECT * FROM evaluations ORDER BY created_at DESC")]


def get_eval(eid):
    with _conn() as c:
        r = c.execute("SELECT * FROM evaluations WHERE id=?", (eid,)).fetchone()
    return _row(r) if r else None


def next_id():
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        i = 1001 + n
        while c.execute("SELECT 1 FROM evaluations WHERE id=?", (f"EV-{i}",)).fetchone():
            i += 1
    return f"EV-{i}"


def save_eval(e):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (e["id"], e["name"], e["created_at"].isoformat(), e["prompt"], e["response"],
                   e.get("expected", ""), e.get("sources", ""), json.dumps(e["scores"]),
                   json.dumps(e["feedback"]), e["mode"]))


def delete_eval(eid):
    with _conn() as c:
        c.execute("DELETE FROM evaluations WHERE id=?", (eid,))


def clear_all():
    with _conn() as c:
        c.execute("DELETE FROM evaluations")


def seed_samples(c=None):
    rnd, names = random.Random(7), ["Refund policy answer", "Python sorting explainer", "Travel itinerary draft",
        "Medical FAQ summary", "SQL query helper", "Product description", "Legal clause summary",
        "Onboarding email", "Climate Q&A", "Tax deadline reply"]
    own = c is None
    c = c or _conn()
    for i in range(30):
        b = 5.5 + rnd.random() * 3.5 + i / 30
        sc = {k: clamp(b + (rnd.random() - .5) * 3 - (.8 if k == "factual" else 0)) for k in DIMS}
        ts = datetime.now() - timedelta(days=89 - i * 3, hours=rnd.randint(0, 20))
        c.execute("INSERT OR REPLACE INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (f"EV-{1001 + i}", f"{names[i % 10]} #{i + 1}", ts.isoformat(), "Sample prompt",
                   "Sample response (seed data)", "", "", json.dumps(sc), json.dumps(feedback(sc, False)), "sample"))
    if own:
        c.commit(); c.close()
    else:
        c.commit()


# ---------------------------------------------------------------- export
def to_csv(evals):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "Name", "Date", "Relevance", "Completeness", "Clarity", "Factual Support", "Overall %", "Status", "Mode"])
    for e in evals:
        w.writerow([e["id"], e["name"], e["created_at"].date().isoformat(), *[e["scores"][k] for k in DIMS],
                    e["overall"], e["status"], e["mode"]])
    return buf.getvalue().encode()


def to_pdf(evals, title="Evaluation Report"):
    """Detailed PDF: summary table, then one section per evaluation."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from xml.sax.saxutils import escape
    st, buf = getSampleStyleSheet(), io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    P = lambda t, s="BodyText": Paragraph(escape(str(t)).replace("\n", "<br/>"), st[s])
    el = [P(title, "Title"), P(f"{len(evals)} evaluation(s). Generated {datetime.now():%Y-%m-%d %H:%M}.")]
    if evals:
        avg = round(sum(e["overall"] for e in evals) / len(evals))
        el.append(P(f"Average overall score: {avg}/100 ({status(avg)})"))
        rows = [["ID", "Name", "Rel", "Com", "Cla", "Fac", "Overall", "Status"]] + [
            [e["id"], e["name"][:28], *[e["scores"][k] for k in DIMS], e["overall"], e["status"]] for e in evals[:60]]
        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), .25, colors.grey)]))
        el += [Spacer(1, 10), t]
    for e in [x for x in evals if x["mode"] != "sample"][:25]:
        el += [PageBreak(), P(f"{e['id']} - {e['name']}", "Heading2"),
               P(f"Overall: {e['overall']}/100 ({e['status']}) | Mode: {e['mode']} | {e['created_at']:%Y-%m-%d}")]
        for k in DIMS:
            el.append(P(f"{LABELS[k]}: {e['scores'][k]}/10 - {e['feedback']['why'][k]}"))
        for h, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"), ("Recommendations", "recs")):
            el += [P(h, "Heading3")] + [P("- " + s) for s in e["feedback"][key]]
        el += [P("Prompt", "Heading3"), P(e["prompt"][:1500]), P("Response", "Heading3"), P(e["response"][:3000])]
    doc.build(el)
    return buf.getvalue()
