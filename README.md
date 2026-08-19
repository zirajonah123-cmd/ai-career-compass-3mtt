# AI Career Compass — 3MTT AI/ML Capstone

**Track:** AI/Machine Learning  
**Project type:** Recommendation / classification prototype

## Problem
Many students know what they study but are unsure which technology career best matches their skills, interests and goals.

## Solution
AI Career Compass is an ML-powered web application that takes a learner's education level, field of study, skills, interests and career goal and ranks suitable career directions. It also provides a skill roadmap and opportunity directions.

## ML pipeline
1. Combine learner profile fields into text.
2. Convert text into TF-IDF features.
3. Classify the profile with multiclass Logistic Regression.
4. Return the top three ranked career classes.
5. Provide a human-readable learning roadmap.

## Technology
- Python
- pandas
- scikit-learn
- joblib
- Streamlit

## Files
- `app.py` — Streamlit application
- `train_model.py` — model training/evaluation
- `model.joblib` — trained model
- `data/career_data.csv` — starter dataset
- `requirements.txt` — dependencies

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Evaluation note
The current starter dataset is synthetic and intentionally small for a learning prototype. Its high test score should **not** be presented as evidence of real-world performance. For the final capstone, the dataset should be expanded with properly sourced data, separated train/test data, cross-validation, confusion matrix, class balance checks, and a documented fairness/limitations assessment.

## Deployment
The app can be deployed through Streamlit Community Cloud after the project folder is pushed to a GitHub repository. The public deployment URL can then be used as the Capstone Project link.

## Ethical considerations
- Do not collect unnecessary personal information.
- Do not present recommendations as guaranteed career outcomes.
- Document dataset sources and consent/licensing where applicable.
- Test performance across relevant groups before real-world use.
