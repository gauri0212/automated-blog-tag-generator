import streamlit as st
import pickle
from utils import preprocess
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Premium Blog Tag Generator",
    page_icon="🚀",
    layout="wide"
)

# ---------- DARK MODE TOGGLE ----------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# ---------- CUSTOM CSS ----------
bg_color = "#0E1117" if dark_mode else "#F5F7FB"
text_color = "white" if dark_mode else "#222"

st.markdown("""
<style>

/* ================== GLOBAL BACKGROUND ================== */
.stApp {
    background: linear-gradient(120deg, #eef2ff, #fdf2f8, #ecfeff);
    background-size: 300% 300%;
    animation: gradientMove 18s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ================== FLOATING BLOBS ================== */
.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.35;
    z-index: 0;
}

.stApp::before {
    background: #6366f1;
    top: -120px;
    left: -120px;
}

.stApp::after {
    background: #22c55e;
    bottom: -140px;
    right: -120px;
}

/* ================== MAIN TITLE ================== */
.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 6px;
    letter-spacing: -0.6px;
}

/* ================== SUBTITLE ================== */
.subtitle {
    text-align: center;
    opacity: 0.75;
    margin-bottom: 34px;
    font-size: 17px;
}

/* ================== GLASS CARD ================== */
.glass-card {
    padding: 30px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 18px 45px rgba(0,0,0,0.12);
    border: 1px solid rgba(255,255,255,0.5);
    position: relative;
    z-index: 1;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 28px 60px rgba(0,0,0,0.18);
}

/* ================== TAG CHIP ================== */
.tag {
    display: inline-block;
    padding: 9px 20px;
    margin: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22c55e);
    color: white;
    font-weight: 700;
    font-size: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    animation: popIn 0.35s ease;
}

@keyframes popIn {
    from {
        transform: scale(0.85);
        opacity: 0;
    }
    to {
        transform: scale(1);
        opacity: 1;
    }
}

/* ================== PREMIUM BUTTON ================== */
.stButton > button {
    border-radius: 12px;
    padding: 0.55rem 1.4rem;
    font-weight: 700;
    border: none;
    background: linear-gradient(90deg, #6366f1, #22c55e);
    color: white;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(99,102,241,0.45);
}

/* ================== TEXT AREA POLISH ================== */
textarea {
    border-radius: 14px !important;
}

/* ================== SECTION SPACING ================== */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown(
    '<div class="main-title">🚀 Premium Blog Tag Generator</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Smart NLP-powered automatic tagging system</div>',
    unsafe_allow_html=True
)

# ---------- LOAD MODELS ----------
try:
    model = pickle.load(open("models/model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    mlb = pickle.load(open("models/mlb.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run train.py first.")
    st.stop()

# ---------- PREDICT FUNCTION ----------
def predict_tags(text, threshold=0.6):
    text = preprocess(text)
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]

    tags = []
    confidences = []

    for i, p in enumerate(probs):
        if p >= threshold:
            tags.append(mlb.classes_[i])
            confidences.append(round(p * 100, 1))

    return tags, confidences

# ---------- LAYOUT ----------
left, right = st.columns([2,1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("✍️ Enter Blog Content")
    blog_text = st.text_area("", height=220)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧪 Load Example"):
            st.session_state.example = (
                "Artificial intelligence is transforming healthcare "
                "and improving medical diagnosis using machine learning."
            )
            st.rerun()

    with col2:
        generate = st.button("🚀 Generate Tags")

    st.markdown('</div>', unsafe_allow_html=True)

    # Apply example if clicked
    if "example" in st.session_state:
        blog_text = st.session_state.example

with right:
    st.subheader("📁 Upload File")

    uploaded_file = st.file_uploader(
        "Upload TXT file",
        type=["txt"]
    )

    # ---------- SAFE FILE READ ----------
if uploaded_file is not None:
    raw_bytes = uploaded_file.read()

    try:
        blog_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            blog_text = raw_bytes.decode("cp1252")
        except:
            blog_text = raw_bytes.decode("latin-1")

    st.success("✅ File loaded successfully!")

# ---------- OUTPUT ----------
if generate:
    if blog_text.strip() == "":
        st.warning("⚠️ Please enter blog content.")
    else:
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        tags, confidences = predict_tags(blog_text)

        st.subheader("🏷️ Suggested Tags")

        if len(tags) == 0:
            st.info("No strong tags detected. Try longer text.")
        else:
            for tag, conf in zip(tags, confidences):
                st.markdown(
                    f'<span class="tag">{tag}</span>',
                    unsafe_allow_html=True
                )
                st.progress(conf / 100)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("✨ Premium NLP Project | Built with Streamlit")