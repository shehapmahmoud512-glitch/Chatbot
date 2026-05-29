from flask import Flask, request, jsonify, send_from_directory, render_template, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
import os
import json
from datetime import timedelta

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get('SECRET_KEY', 'sphinx_university_super_secret_key')

# Use DATABASE_URL from environment if available (useful for Railway Postgres), fallback to local SQLite
db_url = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app, supports_credentials=True)

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    chats = db.relationship('Chat', backref='user', lazy=True)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(100), default="default", nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_reply = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# --------------------------
# 🔹 Knowledge Base Models
# --------------------------

# --------------------------
# 🔹 Gemini Setup (Environment Variables)
# --------------------------
# Get API keys from environment variable (comma-separated) or use defaults
ENV_KEYS = os.environ.get("GEMINI_API_KEYS", "").split(",")
API_KEYS = [k.strip() for k in ENV_KEYS if k.strip()]

MODEL_VERSIONS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

model_idx = 0

# --------------------------
# 🔹 Load Knowledge Base (Dynamic RAG-style)
# --------------------------
CHATBOT_CONTEXT = ""
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "chatbot.txt")

try:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        CHATBOT_CONTEXT = f.read()
except Exception as e:
    print(f"Warning: Could not read chatbot.txt: {e}")

if not CHATBOT_CONTEXT.strip():
    CHATBOT_CONTEXT = "University Information: Sphinx University uses a credit hour system."

# --------------------------
# 🔹 University Rules
# --------------------------
UNIVERSITY_RULES = """
Sphinx University Graduation Requirements:
- Total credit hours required: 138
- Minimum GPA: 2.0 (on a 4.0 scale)
- Minimum attendance rate: 75%
- Maximum study duration: 8 years (16 semesters)
- Students must pass all core/mandatory courses
- Students on academic probation must achieve GPA ≥ 2.0 next semester
- A student fails a course if attendance drops below 75%
- Failed courses can be retaken (counted toward max duration)
"""

# --------------------------
# 🔹 System Prompt
# --------------------------
SYSTEM_INSTRUCTION = f"""You are a friendly human academic advisor for Sphinx University.
You have access to the following university knowledge base and rules.

=== UNIVERSITY KNOWLEDGE BASE ===
{CHATBOT_CONTEXT}

=== GRADUATION RULES ===
{UNIVERSITY_RULES}

=== YOUR BEHAVIOR ===
1. You must first ALWAYS check if the answer exists in the UNIVERSITY KNOWLEDGE BASE or GRADUATION RULES.
2. IF the answer depends on the provided university rules or context, you MUST start your reply exactly with "📚 (من اللائحة): " and provide the rule directly in conversational human text.
3. IF the answer is NOT in the rules/context, you MUST start your reply exactly with "🤖 (AI): " and answer from your general knowledge.
4. DO NOT use any markdown formatting (no asterisks **, no hash #). DO NOT use bullet points or numbered lists.
5. DO NOT output any code or JSON. Speak exactly like a normal person chatting on WhatsApp.
6. If the student shares their academic data (credit hours, GPA, attendance), calculate their eligibility naturally in conversation and prefix with "📊 (تحليل البيانات): ".
7. Respond in the same language the student uses (Arabic or English), and be extremely empathetic and warm."""

# --------------------------
# 🔹 Gemini Setup (Rotating Models & Cache)
# --------------------------
def get_next_model():
    global model_idx
    if not API_KEYS:
        raise ValueError("No API keys configured.")
        
    k_idx = (model_idx // len(MODEL_VERSIONS)) % len(API_KEYS)
    m_idx = model_idx % len(MODEL_VERSIONS)
    
    genai.configure(api_key=API_KEYS[k_idx])
    model_name = MODEL_VERSIONS[m_idx]
    
    return genai.GenerativeModel(model_name, system_instruction=SYSTEM_INSTRUCTION)

try:
    model = get_next_model()
except Exception as e:
    print(f"Warning: Could not initialize Gemini model: {e}")
    class MockModel:
        def generate_content(self, *args, **kwargs):
            return type('obj', (object,), {'text': 'AI Error: No API keys configured.'})
        def start_chat(self, *args, **kwargs):
            return type('obj', (object,), {'send_message': lambda msg: type('obj', (object,), {'text': 'AI Error: No API keys configured.'})})
    model = MockModel()

response_cache = {}

# Logic handled via SYSTEM_INSTRUCTION above

# --------------------------
# 🔹 Chat History (per session - in-memory)
# --------------------------
chat_sessions = {}

import difflib

# --------------------------
# 🔹 Similarity Search (Local Data First)
# --------------------------
def find_local_match(user_query):
    if not CHATBOT_CONTEXT:
        return None

    lines = [
        line.strip()
        for line in CHATBOT_CONTEXT.split("\n")
        if line.strip()
    ]

    matches = difflib.get_close_matches(
        user_query,
        lines,
        n=1,
        cutoff=0.3
    )

    if matches:
        return "📚 (من اللائحة): " + matches[0]

    return None
# --------------------------
# 🔹 /chat Endpoint
# --------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "اسم المستخدم وكلمة المرور مطلوبان"}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"error": "اسم المستخدم موجود مسبقاً"}), 400

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    session["user_id"] = new_user.id
    session["username"] = new_user.username

    return jsonify({"success": True, "message": "تم التسجيل بنجاح", "username": new_user.username})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "اسم المستخدم وكلمة المرور مطلوبان"}), 400

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        session["user_id"] = user.id
        session["username"] = user.username
        return jsonify({"success": True, "message": "تم تسجيل الدخول بنجاح", "username": user.username})
    else:
        return jsonify({"error": "اسم المستخدم أو كلمة المرور غير صحيحة"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return jsonify({"success": True, "message": "تم تسجيل الخروج"})

@app.route("/get_chats", methods=["GET"])
def get_chats():
    if "user_id" not in session:
        return jsonify({"error": "غير مصرح لك"}), 401
    
    user_id = session["user_id"]
    chats = Chat.query.filter_by(user_id=user_id).order_by(Chat.timestamp.asc()).all()
    
    sessions_dict = {}
    for chat in chats:
        s_id = chat.session_id
        if s_id not in sessions_dict:
            title = chat.user_message[:30] + "..." if len(chat.user_message) > 30 else chat.user_message
            sessions_dict[s_id] = {
                "session_id": s_id,
                "title": title,
                "messages": []
            }
        sessions_dict[s_id]["messages"].append({"sender": "user", "text": chat.user_message})
        sessions_dict[s_id]["messages"].append({"sender": "bot", "text": chat.bot_reply})
        
    sessions_list = list(sessions_dict.values())
    sessions_list.reverse()
        
    return jsonify({"success": True, "sessions": sessions_list, "username": session["username"]})

@app.route("/check_auth", methods=["GET"])
def check_auth():
    if "user_id" in session:
        return jsonify({"logged_in": True, "username": session["username"]})
    return jsonify({"logged_in": False})

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"error": "يجب تسجيل الدخول أولاً"}), 401
    data = request.get_json() or {}
    return _handle_chat(data)

def _handle_chat(data):
    global model, model_idx
    try:
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # 1️⃣ FIRST: Try Local Data Match
        local_reply = find_local_match(user_message)
        cache_key = user_message.lower()
        
        bot_reply = None
        source = None

        if local_reply:
            bot_reply = local_reply
            source = "local"
        # 2️⃣ SECOND: Check Global Cache
        elif cache_key in response_cache:
            bot_reply = response_cache[cache_key]
            source = "cache"
        else:
            # 3️⃣ THIRD: Resort to AI
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            
            history = chat_sessions[session_id]
            gemini_history = []
            for turn in history[-5:]:
                gemini_history.append({"role": "user", "parts": [turn["user"]]})
                gemini_history.append({"role": "model", "parts": [turn["bot"]]})

            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_message)
            bot_reply = response.text.strip()
            
            if not bot_reply.startswith("🤖") and not bot_reply.startswith("📚") and not bot_reply.startswith("📊"):
                 bot_reply = "🤖 (AI): " + bot_reply

            # Save to cache
            response_cache[cache_key] = bot_reply
            source = "ai"

        # Save to in-memory history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        chat_sessions[session_id].append({"user": user_message, "bot": bot_reply})

        # Save to database for ALL sources (AI, Cache, Local)
        user_id = session.get("user_id")
        if user_id:
            new_chat = Chat(user_id=user_id, session_id=session_id, user_message=user_message, bot_reply=bot_reply)
            db.session.add(new_chat)
            db.session.commit()
        
        return jsonify({"reply": bot_reply, "session_id": session_id, "source": source})

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Quota" in error_msg or "404" in error_msg:
            retries = data.get("retries", 0)
            if retries >= len(MODEL_VERSIONS) * len(API_KEYS):
                friendly_err = "عذراً، ضغط الأسئلة كبير حالياً على جميع المفاتيح. يرجى الانتظار 30 ثانية والمحاولة مرة أخرى ⏱️"
                return jsonify({"error": friendly_err}), 429
            
            model_idx += 1 
            model = get_next_model()
            data["retries"] = retries + 1
            return _handle_chat(data) 
        
        return jsonify({"error": f"AI Error: {error_msg}"}), 500
# --------------------------
# 🔹 /check-requirements Endpoint
# --------------------------
@app.route("/check-requirements", methods=["POST"])
def check_requirements():
    try:
        data = request.get_json()
        credit_hours = data.get("credit_hours", 0)
        gpa = data.get("gpa", 0.0)
        attendance = data.get("attendance", 0)
        years = data.get("years", 0)
        student_name = data.get("name", "Student")

        prompt = f"""You are a strict academic advisor at Sphinx University.

{UNIVERSITY_RULES}

A student named {student_name} has submitted their academic record for graduation eligibility check:
- Completed Credit Hours: {credit_hours} / 138 required
- Current GPA: {gpa} / 4.0 (minimum required: 2.0)
- Attendance Rate: {attendance}% (minimum required: 75%)
- Years of Study: {years} / 8 maximum

Please provide:
1. ✅ or ❌ for each requirement (pass/fail)
2. An overall verdict: CAN GRADUATE or CANNOT GRADUATE YET
3. If cannot graduate: specific advice on what to improve
4. If can graduate: congratulations and graduation readiness summary
5. Estimated semesters remaining (if applicable)

Be structured, clear, and supportive. Use emojis. Respond in both Arabic and English."""

        response = model.generate_content(prompt)
        result = response.text.strip()

        # Simple pass/fail logic for frontend badge
        can_graduate = (
            int(credit_hours) >= 138 and
            float(gpa) >= 2.0 and
            int(attendance) >= 75 and
            int(years) <= 8
        )

        return jsonify({
            "analysis": result,
            "can_graduate": can_graduate,
            "details": {
                "credit_hours": {"value": credit_hours, "required": 138, "pass": int(credit_hours) >= 138},
                "gpa": {"value": gpa, "required": 2.0, "pass": float(gpa) >= 2.0},
                "attendance": {"value": attendance, "required": 75, "pass": int(attendance) >= 75},
                "years": {"value": years, "required": 8, "pass": int(years) <= 8}
            }
        })

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Quota" in error_msg:
            friendly_err = "عذراً، لقد استنفدت الحد المسموح للذكاء الاصطناعي حالياً. يرجى الانتظار دقيقة والمحاولة."
            return jsonify({"error": friendly_err}), 429
        return jsonify({"error": f"AI Error: {error_msg}"}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    return send_from_directory("static", path)

# --------------------------
# 🔹 Initialize Database & Run
# --------------------------
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=5000)

