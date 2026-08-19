"""
AI Resume Analyzer — Streamlit Dashboard
------------------------------------------------
A clean, modern, HR/AI-styled dashboard that runs a real NLP + ML pipeline
(TF-IDF + Logistic Regression) to classify an uploaded resume into a job
category, scores it for ATS compatibility, and visualizes every stage of
the pipeline.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import os
import re
import time
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------------
# Optional file-reading / NLP libraries (app degrades gracefully if missing)
# ----------------------------------------------------------------------------
try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False

try:
    import docx  # python-docx
    DOCX_OK = True
except Exception:
    DOCX_OK = False

NLTK_OK = True
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    def _ensure_nltk_data():
        packages = {
            "tokenizers/punkt": "punkt",
            "tokenizers/punkt_tab": "punkt_tab",
            "corpora/stopwords": "stopwords",
            "corpora/wordnet": "wordnet",
        }
        for path, pkg in packages.items():
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

    _ensure_nltk_data()
    _ = nltk_stopwords.words("english")
    _lemmatizer = WordNetLemmatizer()
    _ = _lemmatizer.lemmatize("testing")
except Exception:
    NLTK_OK = False

# ----------------------------------------------------------------------------
# Fallback NLP building blocks (used automatically if NLTK data is unavailable)
# ----------------------------------------------------------------------------
FALLBACK_STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

_SUFFIX_RULES = [
    ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
    ("ing", ""), ("edly", ""), ("ed", ""), ("ies", "y"),
    ("es", ""), ("s", ""),
]


def simple_lemmatize(word: str) -> str:
    """Very light rule-based lemmatizer used only if NLTK/WordNet is unavailable."""
    if len(word) <= 3:
        return word
    for suf, repl in _SUFFIX_RULES:
        if word.endswith(suf) and len(word) - len(suf) > 2:
            return word[: -len(suf)] + repl
    return word


def tokenize_text(text: str):
    if NLTK_OK:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)


def remove_stopwords(tokens):
    if NLTK_OK:
        try:
            sw = set(nltk_stopwords.words("english"))
            return [t for t in tokens if t.lower() not in sw and len(t) > 1]
        except Exception:
            pass
    return [t for t in tokens if t.lower() not in FALLBACK_STOPWORDS and len(t) > 1]


def lemmatize_tokens(tokens):
    if NLTK_OK:
        try:
            return [_lemmatizer.lemmatize(t.lower()) for t in tokens]
        except Exception:
            pass
    return [simple_lemmatize(t.lower()) for t in tokens]


def clean_resume_text(txt: str) -> str:
    """Regex-based cleaning identical in spirit to the training pipeline."""
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+|https\S+", " ", txt)
    txt = re.sub(r"@\S+", " ", txt)
    txt = re.sub(r"#\S+", " ", txt)
    txt = re.sub(r"\brt\b", " ", txt)
    txt = re.sub(r"[^a-zA-Z0-9\s\-/._]", " ", txt)
    txt = re.sub(r"[^\x00-\x7f]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# ----------------------------------------------------------------------------
# File extraction
# ----------------------------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        if not PDF_OK:
            st.error("PDF support requires `pdfplumber`. Please install it (see requirements.txt).")
            return ""
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)

    if name.endswith(".docx"):
        if not DOCX_OK:
            st.error("DOCX support requires `python-docx`. Please install it (see requirements.txt).")
            return ""
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    if name.endswith(".txt"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")

    st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return ""


# ----------------------------------------------------------------------------
# Model loading & training (cached — checks pre-trained .pkl first)
# ----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MODEL_EXPORT_PATH = os.path.join(DATA_DIR, "trained_pipeline.pkl")
CSV_CANDIDATES = [
    os.path.join(DATA_DIR, "resume_data.csv"),
    os.path.join(DATA_DIR, "resume_data (1).csv"),
]


def _find_csv_path():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource(show_spinner=False)
def train_pipeline():
    # 1. Attempt to load a pre-trained artifact (tfidf + model + label encoder + centroids)
    if os.path.exists(MODEL_EXPORT_PATH):
        try:
            with open(MODEL_EXPORT_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load pre-trained model ({e}). Training from CSV instead...")

    # 2. Fallback: train live from the CSV dataset
    target_csv = _find_csv_path()
    if target_csv is None:
        st.error(
            f"Missing dataset in `{DATA_DIR}` (expected `resume_data.csv`) and no pre-trained artifact found."
        )
        st.stop()

    df = pd.read_csv(target_csv)
    df = df.dropna(subset=["Category", "Resume"]).reset_index(drop=True)
    df["cleaned"] = df["Resume"].apply(clean_resume_text)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        stop_words="english", max_features=5000, ngram_range=(1, 2), sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test_vec))

    # Category centroids for cosine similarity
    X_full_vec = tfidf.transform(df["cleaned"])
    centroids = {}
    for idx, cls_name in enumerate(label_encoder.classes_):
        mask = y == idx
        centroids[cls_name] = np.asarray(X_full_vec[mask].mean(axis=0))

    return {
        "tfidf": tfidf,
        "model": model,
        "label_encoder": label_encoder,
        "centroids": centroids,
        "test_accuracy": test_accuracy,
        "n_samples": len(df),
        "n_categories": len(label_encoder.classes_),
    }


def analyze_resume(raw_text: str, pipeline: dict, top_n: int = 5):
    tfidf = pipeline["tfidf"]
    model = pipeline["model"]
    le = pipeline["label_encoder"]
    centroids = pipeline["centroids"]

    cleaned = clean_resume_text(raw_text)
    vec = tfidf.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    order = np.argsort(probs)[::-1]
    top_categories = [(le.classes_[i], float(probs[i]) * 100) for i in order[:top_n]]
    predicted_category, confidence = top_categories[0]

    feature_names = tfidf.get_feature_names_out()
    row = vec.toarray()[0]
    nz = np.nonzero(row)[0]
    keywords = sorted([(feature_names[i], float(row[i])) for i in nz], key=lambda x: -x[1])[:15]

    similarities = []
    for cat, centroid in centroids.items():
        sim = float(cosine_similarity(vec, centroid)[0][0]) * 100
        similarities.append((cat, sim))
    similarities.sort(key=lambda x: -x[1])

    return {
        "cleaned_regex": cleaned,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "top_categories": top_categories,
        "keywords": keywords,
        "similarities": similarities[:5],
    }


# ----------------------------------------------------------------------------
# ATS (Applicant Tracking System) scoring
# ----------------------------------------------------------------------------
SKILLS_DB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "php", "ruby", "go", "rust",
    "sql", "mysql", "postgresql", "mongodb", "oracle", "nosql", "sql server",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask", "spring", "spring boot",
    "asp.net", "rails", "laravel", "bootstrap", "jquery",
    "machine learning", "deep learning", "nlp", "computer vision", "data science", "data analysis",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "matplotlib", "seaborn",
    "tableau", "power bi", "excel", "data visualization", "statistics",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "linux", "bash", "terraform",
    "ansible", "prometheus", "grafana", "ci/cd", "devops",
    "agile", "scrum", "project management", "leadership", "team management", "communication",
    "problem solving", "critical thinking", "time management", "collaboration", "decision making",
    "cybersecurity", "network security", "penetration testing", "ethical hacking", "security",
    "sap", "salesforce", "erp", "blockchain", "iot", "robotics", "automation", "testing", "qa",
]

ACTION_VERBS = [
    "led", "built", "developed", "designed", "implemented", "improved", "increased",
    "reduced", "created", "managed", "launched", "optimized", "automated", "delivered",
    "achieved", "drove", "spearheaded", "architected", "streamlined", "mentored",
    "collaborated", "analyzed", "deployed", "engineered",
]

SECTION_HEADERS = {
    "summary": [r"summary", r"objective", r"profile"],
    "skills": [r"skills", r"technical\s*skills", r"core\s*competencies"],
    "experience": [r"experience", r"work\s*history", r"employment"],
    "education": [r"education", r"academic"],
    "projects": [r"projects"],
}


def clean_text_for_ats(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str):
    text_lower = text.lower()
    return sorted({skill for skill in SKILLS_DB if skill in text_lower})


def extract_resume_sections(text: str):
    if not text:
        return {}
    lines = text.split("\n")
    sections, current, buffer = {}, "header", []

    def match_header(line):
        line_clean = line.strip().lower().rstrip(":")
        if not line_clean or len(line_clean) > 40:
            return None
        for sec, patterns in SECTION_HEADERS.items():
            for pat in patterns:
                if re.fullmatch(pat, line_clean) or re.match(pat, line_clean):
                    return sec
        return None

    for line in lines:
        matched = match_header(line)
        if matched:
            if buffer:
                sections[current] = sections.get(current, "") + "\n".join(buffer)
            current, buffer = matched, []
        else:
            buffer.append(line)
    if buffer:
        sections[current] = sections.get(current, "") + "\n".join(buffer)
    return sections


def check_resume_formatting(text: str):
    sections = extract_resume_sections(text)
    checks = {
        "has_email": bool(re.search(r"[\w.\-]+@[\w.\-]+\.\w+", text)),
        "has_phone": bool(re.search(r"(\+?\d[\d\-\s\(\)]{8,}\d)", text)),
        "has_skills_section": "skills" in sections,
        "has_experience_section": "experience" in sections,
        "has_education_section": "education" in sections,
    }
    word_count = len(text.split())
    checks["word_count"] = word_count
    if word_count < 150:
        checks["length_flag"] = "too_short"
    elif word_count > 1200:
        checks["length_flag"] = "too_long"
    else:
        checks["length_flag"] = "ok"
    checks["bullet_count"] = len(re.findall(r"^\s*[\-\u2022\*]", text, re.MULTILINE))
    return checks


def check_action_verbs_and_metrics(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullet_lines = [l for l in lines if re.match(r"^[\-\u2022\*]", l) or len(l.split()) < 25]

    action_verb_count, quantified_count = 0, 0
    for line in bullet_lines:
        clean_line = re.sub(r"^[\-\u2022\*]\s*", "", line).strip().lower()
        first_word = clean_line.split(" ")[0] if clean_line else ""
        if first_word in ACTION_VERBS:
            action_verb_count += 1
        if re.search(r"\d+%|\$\d+|\d+\+|\b\d{2,}\b", line):
            quantified_count += 1

    n = len(bullet_lines)
    return {
        "total_bullets_checked": n,
        "action_verb_ratio": round(action_verb_count / n, 2) if n else 0.0,
        "quantified_ratio": round(quantified_count / n, 2) if n else 0.0,
    }


def extract_experience_years(text: str) -> int:
    if not text:
        return 0
    matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text, re.IGNORECASE)
    return max(map(int, matches)) if matches else 0


def check_experience_match(resume_text: str, job_description: str):
    candidate_years = extract_experience_years(resume_text)
    required_years = extract_experience_years(job_description)
    if required_years == 0:
        status = "not_specified"
    elif candidate_years >= required_years:
        status = "meets_requirement"
    else:
        status = "below_requirement"
    return {"candidate_years": candidate_years, "required_years": required_years, "status": status}


def calculate_ats_score(resume_text: str, job_description: str, tfidf):
    """Computes an ATS-style compatibility score.
    If a job description is supplied, scores similarity + skill overlap against it.
    Otherwise scores general resume health (skills, structure, action verbs, metrics).
    """
    jd_provided = bool(job_description and job_description.strip())

    clean_resume = clean_text_for_ats(resume_text)
    resume_skills = set(extract_skills(clean_resume))
    formatting = check_resume_formatting(resume_text)
    verb_metrics = check_action_verbs_and_metrics(resume_text)

    similarity = None
    matched_skills, missing_skills = set(), set()
    skill_match_pct = 0.0
    experience = None

    if jd_provided:
        clean_jd = clean_text_for_ats(job_description)
        try:
            resume_vec = tfidf.transform([clean_resume])
            jd_vec = tfidf.transform([clean_jd])
            similarity = float(cosine_similarity(resume_vec, jd_vec)[0][0]) * 100
        except Exception:
            similarity = 0.0

        jd_skills = set(extract_skills(clean_jd))
        matched_skills = resume_skills & jd_skills
        missing_skills = jd_skills - resume_skills
        skill_match_pct = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 100.0
        experience = check_experience_match(resume_text, job_description)

        overall = (
            similarity * 0.35
            + skill_match_pct * 0.35
            + verb_metrics["action_verb_ratio"] * 100 * 0.15
            + verb_metrics["quantified_ratio"] * 100 * 0.15
        )
        breakdown = {
            "Similarity to JD": round(similarity, 1),
            "Skill Match": round(skill_match_pct, 1),
            "Action Verbs": round(verb_metrics["action_verb_ratio"] * 100, 1),
            "Quantified Bullets": round(verb_metrics["quantified_ratio"] * 100, 1),
        }
    else:
        matched_skills = resume_skills
        skill_match_pct = min(100.0, len(resume_skills) * 8)
        fmt_score = sum(
            20
            for k in ("has_email", "has_phone", "has_skills_section", "has_experience_section", "has_education_section")
            if formatting[k]
        )
        overall = (
            skill_match_pct * 0.30
            + fmt_score * 0.30
            + verb_metrics["action_verb_ratio"] * 100 * 0.20
            + verb_metrics["quantified_ratio"] * 100 * 0.20
        )
        breakdown = {
            "Skills Detected": round(skill_match_pct, 1),
            "Formatting": round(fmt_score, 1),
            "Action Verbs": round(verb_metrics["action_verb_ratio"] * 100, 1),
            "Quantified Bullets": round(verb_metrics["quantified_ratio"] * 100, 1),
        }

    overall = round(min(100.0, max(0.0, overall)), 1)

    # Feedback tips
    feedback = []
    if jd_provided:
        if missing_skills:
            feedback.append(
                f"Add these missing skills if you have them: {', '.join(sorted(missing_skills)[:8])}."
            )
        if similarity is not None and similarity < 40:
            feedback.append("Your resume's overall wording overlaps little with the job description — mirror its key terms where honest.")
        if experience and experience["status"] == "below_requirement":
            feedback.append(
                f"The role asks for {experience['required_years']}+ years; your resume shows {experience['candidate_years']}."
            )
    if not formatting["has_email"] or not formatting["has_phone"]:
        feedback.append("Make sure your contact info (email & phone) is clearly visible near the top.")
    if not formatting["has_skills_section"]:
        feedback.append("Add a dedicated 'Skills' section — ATS systems weight this heavily.")
    if formatting["length_flag"] == "too_short":
        feedback.append("Your resume looks short — consider adding more detail on projects/experience.")
    elif formatting["length_flag"] == "too_long":
        feedback.append("Your resume looks long — consider trimming to 1-2 pages of the most relevant content.")
    if verb_metrics["action_verb_ratio"] < 0.3:
        feedback.append("Start more bullet points with strong action verbs (e.g. 'Led', 'Built', 'Improved').")
    if verb_metrics["quantified_ratio"] < 0.3:
        feedback.append("Quantify your achievements with numbers/percentages where possible (e.g. 'improved performance by 30%').")
    if not feedback:
        feedback.append("Great job — your resume looks well structured and ATS-friendly!")

    return {
        "overall_score": overall,
        "breakdown": breakdown,
        "jd_provided": jd_provided,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "formatting": formatting,
        "verb_metrics": verb_metrics,
        "experience": experience,
        "feedback": feedback,
    }


# ----------------------------------------------------------------------------
# Streamlit page config + styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #F7F8FC;
}

/* Base text/background colors now come from .streamlit/config.toml's [theme]
   block, so every native widget (labels, dataframes, expanders, alerts,
   dropzones, dropdowns) is consistently colored by Streamlit itself instead
   of being patched here. The rules below only style elements we render
   ourselves via unsafe_allow_html. */

[data-testid="stFileUploaderDropzone"] {
    background: #FAFBFF;
    border: 1.5px dashed #C7CEF0 !important;
    border-radius: 12px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ---- Header ---- */
.app-header {
    padding: 28px 32px;
    border-radius: 20px;
    background: linear-gradient(135deg, #14213D 0%, #2B1B55 100%);
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(20, 33, 61, 0.18);
}
.app-header h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}
.app-header p {
    color: #C7CEF0 !important;
    font-size: 15px;
    margin-top: 6px;
    margin-bottom: 0;
}

/* ---- Cards ---- */
.card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 2px 14px rgba(20, 33, 61, 0.06);
    border: 1px solid #EEF0FA;
    margin-bottom: 20px;
}
.section-title {
    color: #14213D !important;
    font-weight: 700;
    font-size: 20px;
    margin-bottom: 4px;
}
.section-sub {
    color: #6B7280 !important;
    font-size: 13.5px;
    margin-bottom: 16px;
}

/* ---- Result Hero Card ---- */
.result-card {
    background: linear-gradient(135deg, #EEF0FF 0%, #F7F3FF 100%);
    border-radius: 20px;
    padding: 30px 32px;
    border: 1px solid #E1DEFB;
    margin-bottom: 22px;
}
.result-label {
    color: #5B4FE9 !important;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.result-category {
    color: #14213D !important;
    font-size: 34px;
    font-weight: 800;
    margin: 6px 0 2px 0;
}
.result-confidence {
    color: #4F46E5 !important;
    font-size: 16px;
    font-weight: 600;
}

/* ---- ATS score badge ---- */
.ats-score-row { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
.ats-score-circle {
    width: 108px; height: 108px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    font-weight: 800; font-size: 26px;
    color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(20,33,61,0.18);
}
.ats-score-circle small { font-size: 11px; font-weight: 600; opacity: 0.85; }
.ats-good { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
.ats-mid  { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
.ats-low  { background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%); }
.ats-tip {
    background: #F4F5FB;
    border-left: 3px solid #7C3AED;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 13.5px;
    color: #2E3350 !important;
}
.check-item { font-size: 13.5px; margin-bottom: 6px; color: #2E3350 !important; }

/* ---- Pipeline Steps ---- */
.pipeline-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
}
.pipe-step {
    flex: 1;
    min-width: 110px;
    text-align: center;
    padding: 14px 6px;
    border-radius: 14px;
    background: #F4F5FB;
    border: 1.5px solid #E7E9F7;
    transition: all 0.2s ease;
}
.pipe-step.done {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-color: transparent;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.28);
}
.pipe-step .pipe-icon { font-size: 20px; display:block; margin-bottom: 4px;}
.pipe-step .pipe-label { font-size: 11.5px; font-weight: 600; color: #4B5066 !important; }
.pipe-step.done .pipe-label { color: #FFFFFF !important; }
.pipe-arrow { color: #C6CAE6; font-size: 18px; padding: 0 2px; }

/* ---- Chips ---- */
.chip {
    display: inline-block;
    background: #EEF0FF;
    color: #4F46E5 !important;
    border-radius: 999px;
    padding: 5px 13px;
    font-size: 12.5px;
    font-weight: 600;
    margin: 4px 5px 4px 0;
    border: 1px solid #E1DEFB;
}
.chip-missing {
    background: #FEF2F2;
    color: #B91C1C !important;
    border: 1px solid #FCD5D5;
}

/* ---- Buttons ---- */
div.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 26px;
    font-weight: 700;
    font-size: 15px;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
}
div.stButton > button:hover {
    color: white !important;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.32);
}
div.stButton > button p { color: white !important; }

textarea,
[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
    background-color: #FFFFFF !important;
    color: #1F2437 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #1F2437 !important;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFF 100%);
    border-right: 1px solid #EEF0FA;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2.2rem;
}
.sidebar-box {
    background: #F7F8FC;
    border: 1px solid #ECEEFA;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 18px;
}
.sidebar-box-title {
    color: #4F46E5 !important;
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    font-size: 13.5px;
    color: #2E3350 !important;
    padding: 4px 0;
    border-bottom: 1px dashed #E7E9F7;
}
.sidebar-stat:last-child { border-bottom: none; }
.sidebar-stat b { color: #14213D !important; }
.sidebar-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #3A4159 !important;
    padding: 5px 0;
}
.sidebar-step .num {
    width: 20px; height: 20px; border-radius: 50%;
    background: #EEF0FF; color: #4F46E5 !important;
    font-size: 11px; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.sidebar-caption { color: #9AA1BD !important; font-size: 12px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>✨ AI Resume Analyzer</h1>
        <p>Upload a resume and let the NLP + Machine Learning pipeline predict the best-fit job category, score ATS compatibility, and show every processing step.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load / train model
# ----------------------------------------------------------------------------
with st.spinner("Loading NLP model pipeline..."):
    pipeline = train_pipeline()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-box">
            <div class="sidebar-box-title">🤖 Model Info</div>
            <div class="sidebar-stat"><span>Training samples</span><b>{pipeline['n_samples']:,}</b></div>
            <div class="sidebar-stat"><span>Job categories</span><b>{pipeline['n_categories']}</b></div>
            <div class="sidebar-stat"><span>Test accuracy</span><b>{pipeline['test_accuracy']*100:.1f}%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-box-title">🧭 Pipeline Stages</div>
            <div class="sidebar-step"><span class="num">1</span> Text Extraction</div>
            <div class="sidebar-step"><span class="num">2</span> Cleaning</div>
            <div class="sidebar-step"><span class="num">3</span> Tokenization</div>
            <div class="sidebar-step"><span class="num">4</span> Stopword Removal</div>
            <div class="sidebar-step"><span class="num">5</span> Lemmatization</div>
            <div class="sidebar-step"><span class="num">6</span> TF-IDF Vectorization</div>
            <div class="sidebar-step"><span class="num">7</span> ML Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-box-title">🎯 ATS Score</div>
            <div class="sidebar-step" style="color:#3A4159;">Paste a job description in the upload panel for a targeted ATS match score — or leave it blank for a general resume health check.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-caption">Built with Streamlit · scikit-learn · NLTK</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Upload section
# ----------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📤 Upload Resume</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Supported formats: PDF, DOCX, TXT</div>',
    unsafe_allow_html=True,
)

col_up, col_btn = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Drop your resume here", type=["pdf", "docx", "txt"], label_visibility="collapsed"
    )
with col_btn:
    st.write("")
    analyze_clicked = st.button("🚀 Upload & Analyze", use_container_width=True)

with st.expander("📋 Add a Job Description for a targeted ATS score (optional)"):
    job_description_input = st.text_area(
        "Job description",
        placeholder="Paste the job description here to score how well this resume matches it...",
        height=140,
        label_visibility="collapsed",
        key="jd_input",
    )
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Pipeline visual
# ----------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("📄", "Text\nExtraction"),
    ("🧹", "Cleaning"),
    ("🔤", "Tokenization"),
    ("🚫", "Stopword\nRemoval"),
    ("🌱", "Lemmatization"),
    ("🔢", "TF-IDF"),
    ("🤖", "ML\nPrediction"),
]


def render_pipeline_html(active_count: int) -> str:
    chips = []
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        done_cls = "done" if i < active_count else ""
        label_html = label.replace("\n", "<br/>")
        chips.append(
            f'<div class="pipe-step {done_cls}"><span class="pipe-icon">{icon}</span>'
            f'<span class="pipe-label">{label_html}</span></div>'
        )
        if i < len(PIPELINE_STEPS) - 1:
            chips.append('<div class="pipe-arrow">→</div>')
    return f'<div class="pipeline-wrap">{"".join(chips)}</div>'


st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔗 Resume Processing Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">How your resume text flows through the NLP + ML pipeline</div>',
    unsafe_allow_html=True,
)
pipeline_placeholder = st.empty()
pipeline_placeholder.markdown(render_pipeline_html(0), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.raw_text = None
    st.session_state.ats_result = None

if analyze_clicked:
    if uploaded_file is None:
        st.warning("Please upload a resume file first.")
    else:
        raw_text = extract_text_from_file(uploaded_file)
        if not raw_text.strip():
            st.error("Could not extract any text from this file. Please try another file.")
        else:
            for step_idx in range(1, len(PIPELINE_STEPS) + 1):
                pipeline_placeholder.markdown(render_pipeline_html(step_idx), unsafe_allow_html=True)
                time.sleep(0.18)

            tokens = tokenize_text(raw_text)
            tokens_no_stop = remove_stopwords(tokens)
            lemmas = lemmatize_tokens(tokens_no_stop)

            result = analyze_resume(raw_text, pipeline)
            result["display_cleaned"] = " ".join(lemmas)
            result["n_tokens"] = len(tokens)
            result["n_tokens_no_stop"] = len(tokens_no_stop)

            ats_result = calculate_ats_score(raw_text, job_description_input, pipeline["tfidf"])

            st.session_state.result = result
            st.session_state.raw_text = raw_text
            st.session_state.ats_result = ats_result
            st.success("Resume processed successfully!")

result = st.session_state.result
ats_result = st.session_state.ats_result

# ----------------------------------------------------------------------------
# Results Display
# ----------------------------------------------------------------------------
if result:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">✨ AI Prediction Result</div>
            <div class="result-category">{result['predicted_category']}</div>
            <div class="result-confidence">Confidence: {result['confidence']:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- ATS Compatibility Score ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 ATS Compatibility Score</div>', unsafe_allow_html=True)
    subtitle = (
        "How well this resume matches the job description you provided"
        if ats_result["jd_provided"]
        else "General ATS health check (paste a job description above for a targeted match score)"
    )
    st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

    score = ats_result["overall_score"]
    score_cls = "ats-good" if score >= 70 else ("ats-mid" if score >= 40 else "ats-low")

    col_score, col_chart = st.columns([1, 2])
    with col_score:
        st.markdown(
            f"""
            <div class="ats-score-row">
                <div class="ats-score-circle {score_cls}">{score:.0f}<small>/ 100</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        for key in ("has_email", "has_phone", "has_skills_section", "has_experience_section", "has_education_section"):
            label = key.replace("has_", "").replace("_", " ").title()
            icon = "✅" if ats_result["formatting"][key] else "❌"
            st.markdown(f'<div class="check-item">{icon} {label}</div>', unsafe_allow_html=True)

    with col_chart:
        breakdown = ats_result["breakdown"]
        fig_ats = go.Figure(
            go.Bar(
                x=list(breakdown.values()),
                y=list(breakdown.keys()),
                orientation="h",
                marker=dict(
                    color=list(breakdown.values()),
                    colorscale=[[0, "#FBBF77"], [1, "#7C3AED"]],
                    showscale=False,
                ),
                text=[f"{v:.0f}%" for v in breakdown.values()],
                textposition="outside",
                textfont=dict(color="#14213D", size=12),
                cliponaxis=False,
                constraintext="none",
            )
        )
        fig_ats.update_layout(
            height=220,
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis=dict(title="Score (%)", range=[0, 118], color="#14213D", tickfont=dict(color="#14213D")),
            yaxis=dict(color="#14213D", tickfont=dict(color="#14213D")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", color="#14213D"),
            showlegend=False,
        )
        st.plotly_chart(fig_ats, use_container_width=True, theme=None)

        if ats_result["jd_provided"] and (ats_result["matched_skills"] or ats_result["missing_skills"]):
            st.markdown("**Matched skills**")
            st.markdown(
                "".join(f'<span class="chip">{s}</span>' for s in ats_result["matched_skills"][:15]) or "—",
                unsafe_allow_html=True,
            )
            if ats_result["missing_skills"]:
                st.markdown("**Missing skills (found in JD, not in resume)**")
                st.markdown(
                    "".join(f'<span class="chip chip-missing">{s}</span>' for s in ats_result["missing_skills"][:15]),
                    unsafe_allow_html=True,
                )

    st.markdown("**Feedback & Tips**")
    for tip in ats_result["feedback"]:
        st.markdown(f'<div class="ats-tip">💡 {tip}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Top categories + keywords row ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏆 Top Matching Categories</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Highest-probability job categories for this resume</div>', unsafe_allow_html=True)

        cats = [c for c, _ in result["top_categories"]]
        vals = [v for _, v in result["top_categories"]]
        fig = go.Figure(
            go.Bar(
                x=vals,
                y=cats,
                orientation="h",
                marker=dict(
                    color=vals,
                    colorscale=[[0, "#A5B4FC"], [1, "#5B4FE9"]],
                ),
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                textfont=dict(color="#14213D", size=12),
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis=dict(title="Confidence (%)", range=[0, max(vals) * 1.25], color="#14213D", tickfont=dict(color="#14213D")),
            yaxis=dict(autorange="reversed", color="#14213D", tickfont=dict(color="#14213D")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", color="#14213D"),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔑 Important Keywords</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Highest-weighted terms detected in the resume</div>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="chip">{kw}</span>' for kw, _ in result["keywords"][:15])
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TF-IDF scores + Similarity scores row ---
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔢 TF-IDF Scores</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Top terms and their TF-IDF weight</div>', unsafe_allow_html=True)
        tfidf_df = pd.DataFrame(result["keywords"], columns=["Term", "TF-IDF Score"])
        tfidf_df["TF-IDF Score"] = tfidf_df["TF-IDF Score"].round(4)
        st.dataframe(tfidf_df, use_container_width=True, hide_index=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Similarity Scores</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Cosine similarity to each category profile</div>', unsafe_allow_html=True)
        sim_df = pd.DataFrame(result["similarities"], columns=["Category", "Similarity (%)"])
        sim_df["Similarity (%)"] = sim_df["Similarity (%)"].round(2)
        st.dataframe(sim_df, use_container_width=True, hide_index=True, height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Text comparison ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Original vs. Cleaned Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Side-by-side comparison after cleaning, tokenization, stopword removal & lemmatization</div>', unsafe_allow_html=True)

    col_orig, col_clean = st.columns(2)
    with col_orig:
        st.markdown("**Original Resume Text**")
        st.text_area(
            "Original",
            value=st.session_state.raw_text[:5000],
            height=260,
            label_visibility="collapsed",
        )
    with col_clean:
        st.markdown("**Cleaned & Processed Text**")
        st.text_area(
            "Cleaned",
            value=result["display_cleaned"][:5000],
            height=260,
            label_visibility="collapsed",
        )
    st.caption(
        f"Tokens extracted: {result['n_tokens']} → after stopword removal: {result['n_tokens_no_stop']}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Upload a resume and click **Upload & Analyze** to see the AI-powered results.")