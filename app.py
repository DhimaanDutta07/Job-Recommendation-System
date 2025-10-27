from flask import Flask, request, render_template
import pandas as pd
import fitz
import docx
import joblib
import os
import math

app = Flask(__name__)

jobs_df = pd.read_csv("jobs.csv")
jobs_df.columns = jobs_df.columns.str.strip().str.lower()

model = joblib.load("job_model.pkl")
vectorizer = joblib.load("skills_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_text_from_file(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.lower()

def extract_skills(text):
    skill_set = set(jobs_df['skill_required'].dropna().unique())
    found = [skill for skill in skill_set if skill.lower() in text]
    return found

@app.route("/", methods=["GET", "POST"])
def home():
    jobs = None
    job_title = None
    page = int(request.args.get("page", 1))
    per_page = 20
    total_pages = math.ceil(len(jobs_df) / per_page)
    paginated_jobs = jobs_df.iloc[(page - 1) * per_page : page * per_page]

    if request.method == "POST":
        file = request.files.get("cv")
        if file:
            file_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(file_path)
            text = extract_text_from_file(file_path)
            skills = extract_skills(text)
            if skills:
                skills_text = " ".join(skills)
                x_vec = vectorizer.transform([skills_text])
                pred = model.predict(x_vec)
                job_title = label_encoder.inverse_transform(pred)[0]
                jobs = jobs_df[jobs_df["skill_required"].isin(skills)]
            else:
                jobs = pd.DataFrame()

    return render_template(
        "x.html",
        jobs=jobs,
        job_title=job_title,
        paginated_jobs=paginated_jobs,
        page=page,
        total_pages=total_pages
    )

if __name__ == "__main__":
    app.run(debug=True)
