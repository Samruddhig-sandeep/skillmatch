from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import re

app = Flask(__name__)

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------
def extract_text_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.lower()


# ---------------------------
# EXTRACT JD REQUIREMENTS
# ---------------------------
def extract_requirements(text):
    sentences = re.split(r"[.\n]", text)

    trigger_words = [
        "experience", "knowledge", "understanding", "familiarity",
        "ability", "skills", "using", "develop", "design",
        "implement", "integrate", "build", "deploy",
        "debug", "maintain", "optimize",
        "analyze", "train", "evaluate", "create", "manage"
    ]

    strong, weak = [], []

    for s in sentences:
        s = s.strip()
        if len(s.split()) < 5:
            continue

        clean = re.sub(r"[^a-zA-Z0-9\s]", "", s)

        if any(t in s for t in trigger_words):
            strong.append(clean)
        else:
            weak.append(clean)

    return list(set(strong)) if strong else list(set(weak[:12]))


# ---------------------------
# EXTRACT MISSING KEYWORDS
# ---------------------------
def extract_keywords(phrases, resume_text):
    resume_words = set(resume_text.split())

    stopwords = {
        "using","with","such","and","or","to","of","for","from",
        "ability","skills","experience","knowledge","understanding",
        "develop","design","implement","maintain","validate",
        "think","logically","problem","learn",
        "guidance","mentorship","candidate",
        "should","have","has","will","can","you","your",
        "work","working","team","environment","opportunity",
        "role","responsibility","plus","strong","ideal"
    }

    keywords = set()

    for phrase in phrases:
        words = [
            w for w in phrase.split()
            if w.isalpha() and len(w) > 3
            and w not in stopwords
            and w not in resume_words
        ]

        if len(words) >= 2:
            keywords.add(" ".join(words[:3]))
        elif len(words) == 1:
            keywords.add(words[0])

    return sorted(keywords)


# ---------------------------
# CORE ATS SCORING FUNCTION
# ---------------------------
def calculate_ats_score(resume_text, jd_text):

    jd_reqs = extract_requirements(jd_text)
    if not jd_reqs:
        return 0, [], []

    jd_emb = model.encode(jd_reqs, convert_to_tensor=True)

    resume_sentences = [
        s.strip() for s in re.split(r"[.\n]", resume_text)
        if len(s.split()) > 5
    ] or [resume_text]

    resume_emb = model.encode(resume_sentences, convert_to_tensor=True)

    matched = 0
    missing = []

    for i, req in enumerate(jd_reqs):
        sim = util.cos_sim(jd_emb[i], resume_emb)
        if sim.max() >= 0.5:
            matched += 1
        else:
            missing.append(req)

    ats_score = round((matched / len(jd_reqs)) * 100, 2)
    keywords = extract_keywords(missing, resume_text)

    return ats_score, missing, keywords


# ---------------------------
# DASHBOARD ROUTE
# ---------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ---------------------------
# SINGLE RESUME MODE
# ---------------------------
@app.route("/personal", methods=["GET", "POST"])
def personal():

    if request.method == "POST":
        resume_pdf = request.files["resume_pdf"]
        jd_pdf = request.files["jd_pdf"]

        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)

        ats_score, missing, keywords = calculate_ats_score(resume_text, jd_text)

        if ats_score >= 70:
            recommendation = "Great match! Your resume is well aligned."
        elif ats_score >= 40:
            recommendation = "It's good but there is still room for improvement."
        else:
            recommendation = "Low match. Improve job-specific skills."

        return render_template(
            "home.html",
            msg=f"ATS Match Score: {ats_score}%",
            recommendation=recommendation,
            missing=keywords[:8]
        )

    return render_template("home.html")


# ---------------------------
# BULK UPLOAD PAGE
# ---------------------------
@app.route("/bulk")
def bulk():
    return render_template("bulk_ranking.html")


# ---------------------------
# BULK RANKING LOGIC
# ---------------------------
@app.route("/rank_resumes", methods=["POST"])
def rank_resumes():

    jd_pdf = request.files["job_description"]
    resume_files = request.files.getlist("resumes")

    if len(resume_files) > 15:
        return render_template("bulk_ranking.html", error="Maximum 15 resumes allowed.")

    jd_text = extract_text_pdf(jd_pdf)

    results = []

    for resume in resume_files:
        resume_text = extract_text_pdf(resume)

        score, _, _ = calculate_ats_score(resume_text, jd_text)

        results.append({
            "resume_name": resume.filename,
            "score": score
        })

    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return render_template("ranking_result.html", results=ranked_results)


if __name__ == "__main__":
    app.run(debug=True)