"""
SkillForge AI — Python Backend (Gemini via google.genai)
Handles all AI calls for:
  1. Task Evaluator
  2. Career Coach
  3. Skill Gap Analyzer
  4. Resume Builder

Setup:
    pip install flask flask-cors google-genai

Run:
    python app.py

Then open http://localhost:5000
"""

import os, json, re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyANuX3l2Low9qN97Az_aQlVqoA5nqhGBZs")  # ← your key
MODEL = "gemini-3.1-flash-lite-preview"
MAX_TOK = 1500

client = genai.Client(api_key=API_KEY)

app = Flask(__name__, static_folder=".")
CORS(app)


# ── Helpers ────────────────────────────────────────────────────────────────────
def ask(system: str, user: str, max_tokens: int = MAX_TOK) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.7,
        ),
    )
    return response.text


def ask_chat(system: str, history: list, max_tokens: int = MAX_TOK) -> str:
    contents = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.7,
        ),
    )
    return response.text


def parse_json(raw: str) -> dict:
    clean = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
    if fence_match:
        clean = fence_match.group(1).strip()
    if not clean.startswith("{"):
        brace_match = re.search(r"\{[\s\S]*\}", clean)
        if brace_match:
            clean = brace_match.group(0)
    return json.loads(clean)


# ── Static ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


# ── 1. Task Evaluator ──────────────────────────────────────────────────────────
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    d = request.json
    if not d:
        return jsonify({"error": "No JSON body received"}), 400
    role = d.get("role", "").strip()
    task = d.get("task", "").strip()
    work = d.get("work", "").strip()
    if not all([role, task, work]):
        return jsonify({"error": "Missing fields: role, task, and work are required"}), 400

    system = ("You are an expert internship task evaluator for SkillForge AI. "
              "Evaluate submissions rigorously but fairly. "
              "Always respond ONLY with valid JSON — no markdown, no preamble, no backticks.")
    user = f"""Role: {role}
Task: {task}
Submission:
{work}

Respond ONLY with this JSON (no fences, no backticks):
{{
  "score": <0-100 integer>,
  "badge": "<Beginner|Intermediate|Advanced>",
  "dimensions": [
    {{"name":"Quality","score":<0-100>}},
    {{"name":"Completeness","score":<0-100>}},
    {{"name":"Creativity","score":<0-100>}},
    {{"name":"Relevance","score":<0-100>}}
  ],
  "headline": "<6-8 word evaluation title>",
  "feedback": "<4-6 sentence detailed feedback with actionable improvements>"
}}"""
    try:
        raw = ask(system, user)
        print(f"[evaluate] Raw: {raw[:200]}")
        return jsonify(parse_json(raw))
    except Exception as e:
        print(f"[evaluate] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── 2. Career Coach ────────────────────────────────────────────────────────────
@app.route("/api/coach", methods=["POST"])
def coach():
    d = request.json
    if not d:
        return jsonify({"error": "No JSON body received"}), 400
    role    = d.get("role", "intern")
    level   = d.get("level", "Student / Fresher")
    history = d.get("history", [])
    if not history:
        return jsonify({"error": "No message provided"}), 400

    system = (f"You are an empathetic, expert AI Career Coach for SkillForge AI. "
              f"The user is a {level} {role}. "
              "Be warm, specific, and actionable. "
              "Keep responses concise (3-5 sentences per point). "
              "Use line breaks for readability. Never use markdown headers.")
    try:
        reply = ask_chat(system, history)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"[coach] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── 3. Skill Gap Analyzer ──────────────────────────────────────────────────────
@app.route("/api/gap", methods=["POST"])
def gap():
    d = request.json
    if not d:
        return jsonify({"error": "No JSON body received"}), 400
    target  = d.get("target", "").strip()
    current = d.get("current", "")
    skills  = d.get("skills", "").strip()
    notes   = d.get("notes", "")
    if not all([target, skills]):
        return jsonify({"error": "Missing required fields: target and skills"}), 400

    system = ("You are an expert career strategist for SkillForge AI. "
              "Analyze skill gaps and create actionable learning plans. "
              "Always respond ONLY with valid JSON — no markdown, no preamble, no backticks.")
    user = f"""Dream Job: {target}
Current Background: {current or 'Not specified'}
Current Skills: {skills}
Notes: {notes or 'None'}

Respond ONLY with this JSON (no fences, no backticks):
{{
  "gaps": [
    {{"skill":"<skill>","current":<0-100>,"required":<60-100>,"status":"<Missing|Weak|Partial|Good>"}},
    {{"skill":"<skill>","current":<0-100>,"required":<60-100>,"status":"<Missing|Weak|Partial|Good>"}},
    {{"skill":"<skill>","current":<0-100>,"required":<60-100>,"status":"<Missing|Weak|Partial|Good>"}},
    {{"skill":"<skill>","current":<0-100>,"required":<60-100>,"status":"<Missing|Weak|Partial|Good>"}}
  ],
  "steps": [
    {{"title":"<step title>","desc":"<2 sentence desc>","tags":["<tag>","<tag>"]}},
    {{"title":"<step title>","desc":"<2 sentence desc>","tags":["<tag>","<tag>"]}},
    {{"title":"<step title>","desc":"<2 sentence desc>","tags":["<tag>","<tag>"]}},
    {{"title":"<step title>","desc":"<2 sentence desc>","tags":["<tag>","<tag>"]}}
  ],
  "summary": "<3-4 sentence overall summary with encouragement>"
}}"""
    try:
        raw = ask(system, user)
        print(f"[gap] Raw: {raw[:200]}")
        return jsonify(parse_json(raw))
    except Exception as e:
        print(f"[gap] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── 4. Resume Builder ──────────────────────────────────────────────────────────
@app.route("/api/resume", methods=["POST"])
def resume():
    info = request.json
    if not info:
        return jsonify({"error": "No JSON body received"}), 400
    name        = info.get("name", "").strip()
    role        = info.get("role", "").strip()
    tech_skills = info.get("techSkills", "").strip()
    if not all([name, role, tech_skills]):
        return jsonify({"error": "Name, Target Role and Technical Skills are required"}), 400

    edu = f"{info.get('eduDegree','')} | {info.get('eduUni','')} | {info.get('eduGpa','')}"
    system = ("You are an elite resume writer for SkillForge AI. "
              "Create 4 distinct resume versions. "
              "Always respond ONLY with valid JSON. No markdown, no preamble, no backticks.")
    user = f"""Name: {name}
Email: {info.get('email','')} | Phone: {info.get('phone','')} | Location: {info.get('location','')}
Links: {info.get('links','')}
Target Role: {role} | Experience: {info.get('expLevel','Student / Fresher')}
About: {info.get('about','')}
Education: {edu}
Technical Skills: {tech_skills}
Soft Skills: {info.get('softSkills','')}
Tools: {info.get('tools','')}
Work: {info.get('work','None')}
Projects: {info.get('projects','')}
Certs: {info.get('certs','')}
Awards: {info.get('awards','')}
Extras: {info.get('extras','')}

Generate 4 styles — classic (formal), modern (tech/startup), creative (bold/storytelling), ats (keyword-heavy).

Respond ONLY with this JSON (no fences, no backticks):
{{
  "classic": {{
    "summary":"<3-sentence formal summary>",
    "experience":[{{"title":"<r>","company":"<c>","period":"<p>","bullets":["<b>","<b>","<b>"]}}],
    "skills":["<s>","<s>","<s>","<s>","<s>","<s>","<s>","<s>"],
    "education":"<deg, uni, year, gpa>","certifications":"<certs>"
  }},
  "modern": {{
    "summary":"<2-sentence punchy summary>",
    "experience":[{{"title":"<r>","company":"<c>","period":"<p>","bullets":["<b>","<b>","<b>"]}}],
    "skills":["<s>","<s>","<s>","<s>","<s>","<s>"],
    "tools":["<t>","<t>","<t>","<t>"],
    "education":"<deg, uni, year>","certifications":"<certs>"
  }},
  "creative": {{
    "tagline":"<bold personal tagline>",
    "summary":"<3-sentence storytelling summary>",
    "experience":[{{"title":"<r>","company":"<c>","period":"<p>","bullets":["<b>","<b>"]}}],
    "skills":["<s>","<s>","<s>","<s>","<s>","<s>"],
    "education":"<deg, uni, year>","certifications":"<certs>"
  }},
  "ats": {{
    "summary":"<3-sentence keyword-rich summary>",
    "experience":[{{"title":"<r>","company":"<c>","period":"<p>","bullets":["<b>","<b>","<b>"]}}],
    "skills":["<s>","<s>","<s>","<s>","<s>","<s>","<s>","<s>","<s>","<s>"],
    "education":"<deg, uni, year, gpa>","certifications":"<all certs>",
    "ats_note":"<why this is ATS optimised>"
  }},
  "ats_tips":"<4-5 sentences of universal resume tips>"
}}"""
    try:
        raw = ask(system, user, max_tokens=2000)
        print(f"[resume] Raw: {raw[:200]}")
        return jsonify(parse_json(raw))
    except Exception as e:
        print(f"[resume] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY or API_KEY == "PASTE_YOUR_KEY_HERE":
        print("\n⚠️  GEMINI_API_KEY not set! Paste your key on line 28.\n")
    else:
        print(f"\n✅ Gemini key loaded ({API_KEY[:12]}...)\n")
    print("🚀 Running at http://localhost:5000\n")
    app.run(debug=True, port=5000)