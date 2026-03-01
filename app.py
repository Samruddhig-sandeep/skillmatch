from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import re

app = Flask(__name__)

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# PDF text extraction
def extract_text_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.lower()

# Extract requirements from JD
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

# Extract keywords
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

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        resume_pdf = request.files["resume_pdf"]
        jd_pdf = request.files["jd_pdf"]

        resume_text = extract_text_pdf(resume_pdf)
        jd_text = extract_text_pdf(jd_pdf)

        jd_reqs = extract_requirements(jd_text)
        if not jd_reqs:
            return render_template("home.html", msg="Not enough data.", missing=[])

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

        if ats_score >= 70:
            recommendation = "Great match! Your resume is well aligned."
        elif ats_score >= 40:
            recommendation = "It's good but there is still room for improvement."
        else:
            recommendation = "Low match. Improve job-specific skills."

        keywords = extract_keywords(missing, resume_text)

        return render_template(
            "home.html",
            msg=f"ATS Match Score: {ats_score}%",
            recommendation=recommendation,
            missing=keywords[:8]
        )

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)