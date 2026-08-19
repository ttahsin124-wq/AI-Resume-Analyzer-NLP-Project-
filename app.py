"""
AI Resume Analyzer — Streamlit Dashboard
------------------------------------------------
A clean, modern, HR/AI-styled dashboard that runs a real NLP + ML pipeline
(TF-IDF + Logistic Regression) to classify an uploaded resume into a job category,
visualizing every stage of the pipeline.

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
CSV_PATH = os.path.join(DATA_DIR, "resume_data (1).csv")


@st.cache_resource(show_spinner=False)
def train_pipeline():
    # 1. Attempt to load pre-trained artifact payload from train_export.py
    if os.path.exists(MODEL_EXPORT_PATH):
        try:
            with open(MODEL_EXPORT_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load pre-trained model ({e}). Re-training from CSV...")

    # 2. Fallback to dynamic training if .pkl is not available
    if not os.path.exists(CSV_PATH):
        # Alternative fallback check for filename variations
        alt_path = os.path.join(DATA_DIR, "resume_data (1).csv")
        target_csv = alt_path if os.path.exists(alt_path) else CSV_PATH
    else:
        target_csv = CSV_PATH

    if not os.path.exists(target_csv):
        st.error(f"Missing dataset at `{target_csv}` and no pre-trained artifact found.")
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

    # Calculate category centroids for cosine similarity
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
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
}
.app-header p {
    color: #C7CEF0;
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
    color: #14213D;
    font-weight: 700;
    font-size: 20px;
    margin-bottom: 4px;
}
.section-sub {
    color: #6B7280;
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
    color: #5B4FE9;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.result-category {
    color: #14213D;
    font-size: 34px;
    font-weight: 800;
    margin: 6px 0 2px 0;
}
.result-confidence {
    color: #4F46E5;
    font-size: 16px;
    font-weight: 600;
}

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
.pipe-step .pipe-label { font-size: 11.5px; font-weight: 600; color: #4B5066; }
.pipe-step.done .pipe-label { color: #FFFFFF; }
.pipe-arrow { color: #C6CAE6; font-size: 18px; padding: 0 2px; }

/* ---- Chips ---- */
.chip {
    display: inline-block;
    background: #EEF0FF;
    color: #4F46E5;
    border-radius: 999px;
    padding: 5px 13px;
    font-size: 12.5px;
    font-weight: 600;
    margin: 4px 5px 4px 0;
    border: 1px solid #E1DEFB;
}

/* ---- Buttons ---- */
div.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 26px;
    font-weight: 700;
    font-size: 15px;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
}
div.stButton > button:hover {
    color: white;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.32);
}

textarea {
    border-radius: 12px !important;
    background: #FAFBFF !important;
    color: #14213D !important;
}

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #EEF0FA;
}
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
        <p>Upload a resume and let the NLP + Machine Learning pipeline predict the best-fit job category — with full transparency into every processing step.</p>
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
    st.markdown("### 🤖 Model Info")
    st.markdown(
        f"""
        - **Algorithm:** TF-IDF + Logistic Regression  
        - **Training samples:** {pipeline['n_samples']:,}  
        - **Job categories:** {pipeline['n_categories']}  
        - **Test accuracy:** {pipeline['test_accuracy']*100:.1f}%
        """
    )
    st.markdown("---")
    st.markdown("### 🧭 Pipeline Stages")
    st.markdown(
        """
        1. Text Extraction
        2. Cleaning
        3. Tokenization
        4. Stopword Removal
        5. Lemmatization
        6. TF-IDF Vectorization
        7. ML Prediction
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · NLTK")

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

            st.session_state.result = result
            st.session_state.raw_text = raw_text
            st.success("Resume processed successfully!")

result = st.session_state.result

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
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis=dict(title="Confidence (%)", range=[0, max(vals) * 1.25]),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter", color="#14213D"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔑 Important Keywords</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Highest-weighted terms detected in the resume</div>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="chip">{kw}</span>' for kw, _ in result["keywords"][:15])
        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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