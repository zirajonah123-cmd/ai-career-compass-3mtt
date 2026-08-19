import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
DATA_PATH = ROOT / "career_data.csv"

st.set_page_config(page_title="AI Career Compass", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

CAREER_INFO = {
    "Data Scientist": {
        "skills": ["Python", "Statistics", "Pandas", "SQL", "Data Visualization"],
        "path": "Python → Data Analysis → Statistics → Machine Learning → Portfolio",
        "opportunities": ["Data internships", "Research assistant roles", "Analytics projects"]
    },
    "Machine Learning Engineer": {
        "skills": ["Python", "Scikit-learn", "Machine Learning", "Algorithms", "Model Deployment"],
        "path": "Python → ML Fundamentals → Model Evaluation → APIs → Cloud Deployment",
        "opportunities": ["AI/ML internships", "Kaggle-style projects", "Junior ML roles"]
    },
    "Data Analyst": {
        "skills": ["Excel", "SQL", "Power BI", "Python", "Data Cleaning"],
        "path": "Excel → SQL → Power BI → Python → Business Case Studies",
        "opportunities": ["Business analytics internships", "Dashboard projects", "Data assistant roles"]
    },
    "Software Developer": {
        "skills": ["Python/JavaScript", "Git", "HTML/CSS", "APIs", "Testing"],
        "path": "Programming → Git → Web/API Development → Testing → Deployment",
        "opportunities": ["Software internships", "Open-source projects", "Junior developer roles"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Networking", "Linux", "Security Fundamentals", "Python", "Risk Assessment"],
        "path": "Networking → Linux → Security Fundamentals → Defensive Labs → SOC Skills",
        "opportunities": ["Cybersecurity labs", "IT support roles", "Security internships"]
    },
    "AI/NLP Specialist": {
        "skills": ["Python", "NLP", "Machine Learning", "Text Processing", "PyTorch"],
        "path": "Python → ML → NLP → Transformers → AI Application Projects",
        "opportunities": ["AI research projects", "NLP prototypes", "AI internships"]
    }
}

st.title("🤖 AI Career Compass")
st.subheader("AI-powered career guidance for students and early-career learners")
st.write(
    "Enter your background and goals. The machine-learning model ranks career directions "
    "and provides a practical learning path."
)

with st.sidebar:
    st.header("About this project")
    st.write("**3MTT Track:** AI/ML")
    st.write("**Technology:** Python + scikit-learn + Streamlit")
    st.write("**Model:** TF-IDF + Logistic Regression")
    st.info(
        "This is an educational prototype. Recommendations are guidance, not a guarantee "
        "of career suitability."
    )

with st.form("career_form"):
    col1, col2 = st.columns(2)
    with col1:
        education = st.selectbox("Education level", ["Diploma", "Undergraduate", "Graduate"])
        field = st.selectbox(
            "Field of study",
            ["Computer Science", "Software Engineering", "Computer Engineering",
             "Data Science", "Statistics", "Mathematics", "Economics",
             "Business Administration", "Cybersecurity", "Information Technology",
             "Linguistics"]
        )
        skills = st.text_input("Skills", placeholder="Python, SQL, statistics")
    with col2:
        interests = st.text_input("Interests", placeholder="AI, data, research")
        goal = st.text_input("Career goal", placeholder="machine learning engineer")
        submitted = st.form_submit_button("🚀 Get AI Recommendation", use_container_width=True)

if submitted:
    text = " | ".join([education, field, skills, interests, goal])
    probabilities = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
    prediction = ranked[0][0]

    st.success(f"### Recommended career: {prediction}")

    c1, c2, c3 = st.columns(3)
    for container, (career, score) in zip([c1, c2, c3], ranked):
        with container:
            st.metric(career, f"{score * 100:.1f}%")

    info = CAREER_INFO[prediction]
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("🎯 Skills to strengthen")
        for skill in info["skills"]:
            st.write(f"• {skill}")
        st.subheader("🛣️ Suggested learning path")
        st.write(info["path"])
    with right:
        st.subheader("💼 Opportunity directions")
        for item in info["opportunities"]:
            st.write(f"• {item}")
        st.subheader("Why this result?")
        st.write(
            "The model compares the text representation of your education, field, skills, "
            "interests and goal with patterns learned from the training examples."
        )

st.divider()
tab1, tab2 = st.tabs(["📊 Model", "📚 Project"])
with tab1:
    df = pd.read_csv(DATA_PATH)
    st.write(f"Starter training records: **{len(df)}**")
    st.write("Career classes:", ", ".join(sorted(df["career"].unique())))
    st.caption("The starter dataset is synthetic and intended for demonstration. A production system should use a larger, representative, ethically sourced dataset.")
with tab2:
    st.write("**Problem:** Students often struggle to connect their current skills and interests to realistic career directions.")
    st.write("**Solution:** An ML recommendation prototype that ranks career paths and gives an actionable skill roadmap.")
    st.write("**Next version:** Add verified Nigerian opportunities, a larger dataset, feedback collection, fairness checks, and stronger evaluation.")

st.caption("3MTT AI/ML Capstone • AI Career Compass")
