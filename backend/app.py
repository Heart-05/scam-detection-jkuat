"""
app.py
AI-Based Scam Message Detection System — JKUAT IT Department
Flask backend implementing all functional requirements from Chapter 3:
  FR1  Message input and submission
  FR2  NLP text preprocessing
  FR3  TF-IDF feature extraction
  FR4  Scam detection and classification
  FR5  Confidence score generation
  FR6  Risk level assignment
  FR7  Keyword-based explanation
  FR8  Display of prediction results
  FR9  Session history
  FR10 Admin monitoring (system logs)
  FR11 Client summary reporting (IT Department)
"""

import re
import uuid
import logging
import sqlite3
import joblib
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, session, g
from flask_cors import CORS

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Secret key is required for Flask session cookies.
# In production replace this with a long random value stored in an env variable.
app.secret_key = "replace-this-with-a-secure-random-key-in-production"

# ── Admin authentication ───────────────────────────────────────────────────────
# Change this token before deployment. Store it securely, not in source code.
ADMIN_TOKEN = "jkuat_admin_2025"

def require_admin_auth():
    """
    Token-based authentication for admin endpoints.
    The client must send the token in the Authorization header:
      Authorization: Bearer jkuat_admin_2025
    This keeps the token out of URLs, browser history, and server logs.
    Returns None if authorised, or a 401 JSON response if not.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header[7:] != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorised. Provide a valid admin token."}), 401
    return None

app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"]   = False
app.config["SESSION_COOKIE_HTTPONLY"] = True

# FR-Security: restrict CORS to the React frontend origin only.
CORS(app, supports_credentials=True, origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
])

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
MODEL_PATH      = BASE_DIR / "model" / "scam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"
DB_PATH         = BASE_DIR / "scam_detection.db"

# ── Model version (stored in ModelRecords table) ───────────────────────────────
MODEL_VERSION = "1.0"
MODEL_NAME    = "Logistic Regression"

# ── Load model and vectorizer once at startup ──────────────────────────────────
model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ── Logging (FR10 — Admin monitoring) ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Database ───────────────────────────────────────────────────────────────────
def get_db():
    """Return a per-request SQLite connection (stored on Flask's g object)."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """
    Create all four tables from the Chapter 3 database schema (Section 3.9.1).
    Called once when the app starts.
    Tables:
      PredictionRecords — one row per prediction (FR9)
      Explanations      — keyword explanation linked to each prediction (FR7)
      ModelRecords      — model version metadata
      SystemLogs        — system events for Admin (FR10)
    """
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS PredictionRecords (
            prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT    NOT NULL,
            result          TEXT    NOT NULL,
            confidence_score REAL   NOT NULL,
            risk_level      TEXT    NOT NULL,
            model_id        INTEGER,
            predicted_at    DATETIME NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS Explanations (
            explanation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id   INTEGER NOT NULL,
            matched_keywords TEXT,
            explanation_text TEXT   NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES PredictionRecords(prediction_id)
        );

        CREATE TABLE IF NOT EXISTS ModelRecords (
            model_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT    NOT NULL,
            version     TEXT    NOT NULL,
            accuracy    REAL    NOT NULL DEFAULT 0.0,
            trained_at  DATETIME NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS SystemLogs (
            log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT,
            prediction_id INTEGER,
            event         TEXT    NOT NULL,
            level         TEXT    NOT NULL DEFAULT 'INFO',
            logged_at     DATETIME NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (prediction_id) REFERENCES PredictionRecords(prediction_id)
        );
    """)

    # Ensure a ModelRecords entry exists for the current model version
    existing = cursor.execute(
        "SELECT model_id FROM ModelRecords WHERE version = ?", (MODEL_VERSION,)
    ).fetchone()
    if not existing:
        cursor.execute(
            "INSERT INTO ModelRecords (model_name, version, accuracy) VALUES (?, ?, ?)",
            (MODEL_NAME, MODEL_VERSION, 0.0)
        )

    db.commit()
    db.close()


def get_model_id():
    """Return the model_id for the current model version."""
    db = get_db()
    row = db.execute(
        "SELECT model_id FROM ModelRecords WHERE version = ?", (MODEL_VERSION,)
    ).fetchone()
    return row["model_id"] if row else None


def log_event(event: str, level: str = "INFO",
              session_id: str = None, prediction_id: int = None):
    """Write a row to SystemLogs and also emit a Python log record (FR10)."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO SystemLogs (session_id, prediction_id, event, level) VALUES (?,?,?,?)",
            (session_id, prediction_id, event, level)
        )
        db.commit()
    except Exception as exc:
        logger.error("Failed to write to SystemLogs: %s", exc)

    if level == "ERROR":
        logger.error(event)
    elif level == "WARNING":
        logger.warning(event)
    else:
        logger.info(event)


# ── JKUAT-specific keyword list (Section 3.9.3c and dataset analysis) ─────────
SCAM_KEYWORDS = [
    # JKUAT / Kenyan student context
    "helb", "scholarship", "bursary", "portal", "hostel", "accommodation",
    "internship", "attachment", "exam results", "missing marks", "fees",
    "school fees", "mpesa", "m-pesa", "mobile money", "safaricom",
    "student loan", "fee balance", "clearance", "registration", "unit registration",
    "examination", "supplementary", "cat results", "timetable", "lecture",
    "university notice", "jkuat", "kuccps", "helb portal", "student portal",

    # Urgency
    "urgent", "act now", "apply now", "call now", "click here",
    "click below", "get it now", "do it today", "limited time",
    "order now", "take action", "selected", "expires", "immediately",
    "respond now", "last chance", "final notice", "action required",
    "within 24 hours", "within 48 hours", "deadline", "today only",

    # Security / credential scams
    "verify", "verification", "password", "passwords", "pin", "otp",
    "account", "blocked", "suspended", "billing", "bank", "credit card",
    "social security number", "confirm your details", "log in", "login",
    "access denied", "confirm identity", "reactivate", "unlock account",
    "update your details", "confirm your account", "security alert",
    "unauthorized access", "suspicious activity",

    # Financial tricks
    "deposit", "send", "transfer", "loan", "loans", "debt",
    "refinance", "mortgage", "rates", "shillings", "kes", "ksh",
    "send money", "paybill", "till number", "stk push", "lipa na mpesa",
    "withdraw", "airtime", "top up", "float",

    # Fake claims
    "no fees", "no cost", "no obligation", "no purchase necessary",
    "hidden charges", "no hidden fees", "risk-free",
    "this isn't a scam", "this isn't spam", "not junk",
    "100% legitimate", "trusted source", "officially approved",

    # Money / reward language
    "free", "cash", "bonus", "prize", "reward", "money", "win",
    "winner", "winning", "giveaway", "guaranteed", "profit",
    "income", "investment", "million", "billion", "refund",
    "congratulations", "you have been selected", "you have won",
    "claim your", "awarded", "lucky winner", "gift", "voucher",

    # Links and phishing
    "link", "http", "www", "click", "bit.ly", "tinyurl", "shortlink",
    "follow the link", "visit the link", "open the link",

    # Impersonation
    "safaricom mpesa", "equity bank", "kcb", "co-operative bank",
    "national government", "ministry of education", "kenya revenue authority",
    "kra", "nhif", "nssf", "county government",

    # Marketing spam
    "special promotion", "exclusive deal", "cheap", "discount",
    "deal", "offer", "trial", "free trial", "work from home",
    "be your own boss", "earn money", "make money", "extra cash",
    "part time job", "online job", "typing job", "data entry job"
]


# ── NLP preprocessing (FR2) ───────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Mirrors train_model.py preprocessing exactly so the vectorizer
    receives text in the same format it was trained on.
    """
    text = str(text).lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", " link ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Keyword extraction (FR7) ──────────────────────────────────────────────────
def extract_keywords(message: str) -> list:
    """Scan the original (non-preprocessed) message for scam indicators."""
    lower = str(message).lower()
    return [kw for kw in SCAM_KEYWORDS if kw in lower]


# ── Explanation generation (FR7) ──────────────────────────────────────────────
def generate_explanation(prediction: str, keywords: list) -> str:
    if prediction == "Scam":
        if keywords:
            return (
                "This message was classified as scam because it contains "
                "suspicious indicators such as: " + ", ".join(keywords) + "."
            )
        return (
            "This message was classified as scam because its text pattern "
            "matches known scam messages, even though no specific keywords "
            "were detected."
        )
    return (
        "This message appears legitimate and does not strongly match "
        "common scam message patterns."
    )


# ── Risk level (FR6) ──────────────────────────────────────────────────────────
def get_risk_level(prediction: str, confidence: float) -> str:
    if prediction == "Scam" and confidence >= 85:
        return "High"
    elif prediction == "Scam" and confidence >= 65:
        return "Medium"
    return "Low"


# ── Session ID helper ─────────────────────────────────────────────────────────
def get_session_id() -> str:
    """
    Return a session ID for the current user.
    Checks X-Session-ID header first (sent by React frontend),
    then falls back to Flask session cookie.
    This avoids SameSite cookie issues in local development.
    """
    # Check header first (React sends this explicitly)
    header_sid = request.headers.get("X-Session-ID", "").strip()
    if header_sid:
        return header_sid
    # Fall back to Flask session cookie
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({"message": "AI Scam Detection System — JKUAT IT Department"})


# ── FR1 / FR2–FR8: Main prediction endpoint ───────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    session_id = get_session_id()

    # FR1 / NFR5: Input validation
    data = request.get_json(silent=True)
    if not data or "message" not in data or not str(data["message"]).strip():
        log_event("Invalid request — missing or empty message field",
                  level="WARNING", session_id=session_id)
        return jsonify({"error": "Message is required"}), 400

    message = str(data["message"]).strip()

    # Optional: limit input length (NFR4 security)
    if len(message) > 5000:
        return jsonify({"error": "Message exceeds maximum length of 5000 characters"}), 400

    try:
        # FR2: NLP preprocessing
        clean_message = preprocess_text(message)

        # FR3: TF-IDF feature extraction
        features = vectorizer.transform([clean_message])

        # FR4: Classification
        prediction_raw = model.predict(features)[0]           # 'scam' or 'legitimate'
        probability     = model.predict_proba(features).max() * 100  # 0–100

        # FR5: Confidence — use the model's actual probability directly (no override)
        confidence = round(probability, 2)

        # FR7: Keyword extraction and explanation
        keywords    = extract_keywords(message)
        prediction  = "Scam" if prediction_raw == "scam" else "Legitimate"
        explanation = generate_explanation(prediction, keywords)

        # FR6: Risk level
        risk_level = get_risk_level(prediction, confidence)

        # Persist to database (FR9, FR10)
        db       = get_db()
        model_id = get_model_id()

        cursor = db.execute(
            """INSERT INTO PredictionRecords
               (session_id, result, confidence_score, risk_level, model_id)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, prediction, confidence, risk_level, model_id)
        )
        prediction_id = cursor.lastrowid

        db.execute(
            """INSERT INTO Explanations
               (prediction_id, matched_keywords, explanation_text)
               VALUES (?, ?, ?)""",
            (prediction_id,
             ", ".join(keywords) if keywords else None,
             explanation)
        )
        db.commit()

        log_event(
            f"Prediction made: {prediction} ({confidence:.1f}%) risk={risk_level}",
            level="INFO", session_id=session_id, prediction_id=prediction_id
        )

        # FR8: Return full result to frontend (include sessionId so React can persist it)
        return jsonify({
            "prediction":    prediction,
            "confidence":    confidence,
            "riskLevel":     risk_level,
            "keywords":      keywords,
            "explanation":   explanation,
            "predictionId":  prediction_id,
            "sessionId":     session_id
        })

    except Exception as exc:
        log_event(f"Prediction error: {exc}", level="ERROR", session_id=session_id)
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


# ── FR9: Session history ──────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history():
    """
    Return prediction records for the current session only.
    Students can only see their own history (NFR9 — privacy).
    """
    session_id = get_session_id()
    try:
        db = get_db()
        rows = db.execute(
            """SELECT p.prediction_id, p.result, p.confidence_score,
                      p.risk_level, p.predicted_at,
                      e.matched_keywords, e.explanation_text
               FROM PredictionRecords p
               LEFT JOIN Explanations e ON e.prediction_id = p.prediction_id
               WHERE p.session_id = ?
               ORDER BY p.predicted_at DESC""",
            (session_id,)
        ).fetchall()

        records = [
            {
                "predictionId":  r["prediction_id"],
                "result":        r["result"],
                "confidence":    r["confidence_score"],
                "riskLevel":     r["risk_level"],
                "predictedAt":   r["predicted_at"],
                "keywords":      r["matched_keywords"].split(", ") if r["matched_keywords"] else [],
                "explanation":   r["explanation_text"]
            }
            for r in rows
        ]
        return jsonify({"history": records, "count": len(records)})

    except Exception as exc:
        log_event(f"History retrieval error: {exc}", level="ERROR", session_id=session_id)
        return jsonify({"error": "Could not retrieve history"}), 500


# ── FR10: Admin — system logs ─────────────────────────────────────────────────
@app.route("/admin/logs", methods=["GET"])
def admin_logs():
    """
    Return recent system log entries for the Admin/System Operator.
    Does NOT expose individual student message content (NFR9).
    Protected by admin key authentication.
    """
    auth_error = require_admin_auth()
    if auth_error:
        return auth_error
    limit = min(int(request.args.get("limit", 100)), 500)
    level = request.args.get("level")   # optional filter: INFO / WARNING / ERROR

    try:
        db = get_db()
        if level:
            rows = db.execute(
                "SELECT * FROM SystemLogs WHERE level = ? ORDER BY logged_at DESC LIMIT ?",
                (level.upper(), limit)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM SystemLogs ORDER BY logged_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

        logs = [dict(r) for r in rows]
        return jsonify({"logs": logs, "count": len(logs)})

    except Exception as exc:
        logger.error("Admin logs error: %s", exc)
        return jsonify({"error": "Could not retrieve logs"}), 500


# ── FR11: IT Department summary report ────────────────────────────────────────
@app.route("/admin/summary", methods=["GET"])
def admin_summary():
    """
    Aggregate statistics for the JKUAT IT Department (FR11).
    Returns counts and top keywords — never exposes individual
    student session IDs or message content.
    Protected by admin key authentication.
    """
    auth_error = require_admin_auth()
    if auth_error:
        return auth_error
    try:
        db = get_db()

        total = db.execute(
            "SELECT COUNT(*) AS n FROM PredictionRecords"
        ).fetchone()["n"]

        scam_count = db.execute(
            "SELECT COUNT(*) AS n FROM PredictionRecords WHERE result = 'Scam'"
        ).fetchone()["n"]

        legitimate_count = db.execute(
            "SELECT COUNT(*) AS n FROM PredictionRecords WHERE result = 'Legitimate'"
        ).fetchone()["n"]

        high_risk = db.execute(
            "SELECT COUNT(*) AS n FROM PredictionRecords WHERE risk_level = 'High'"
        ).fetchone()["n"]

        avg_confidence = db.execute(
            "SELECT AVG(confidence_score) AS avg FROM PredictionRecords"
        ).fetchone()["avg"] or 0.0

        # Top detected keywords (aggregate — no per-student data)
        kw_rows = db.execute(
            "SELECT matched_keywords FROM Explanations WHERE matched_keywords IS NOT NULL"
        ).fetchall()
        keyword_freq = {}
        for row in kw_rows:
            for kw in row["matched_keywords"].split(", "):
                kw = kw.strip()
                if kw:
                    keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            "totalPredictions":    total,
            "scamCount":           scam_count,
            "legitimateCount":     legitimate_count,
            "highRiskCount":       high_risk,
            "averageConfidence":   round(avg_confidence, 2),
            "topKeywords":         [{"keyword": k, "count": v} for k, v in top_keywords],
            "generatedAt":         datetime.utcnow().isoformat() + "Z"
        })

    except Exception as exc:
        logger.error("Summary report error: %s", exc)
        return jsonify({"error": "Could not generate summary"}), 500


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    logger.info("Database initialised. Starting Flask...")
    app.run(debug=True)