import os
import re
import warnings
import pandas as pd
import numpy as np
import pdfplumber
import spacy

import nltk
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

warnings.filterwarnings("ignore")

# Load spaCy English model 
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as spacy_download
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# SECTION 1 - SKILL DATABASE
# (Covers tech, data, cloud, tools, soft skills & domains)

SKILLS_DB = {
    # Programming Languages 
    "python", "java", "c", "c++", "c#", "r", "scala",
    "golang", "rust", "kotlin", "swift", "perl", "ruby",
    "matlab", "bash", "shell scripting",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "oracle",
    "mongodb", "cassandra", "redis", "elasticsearch",
    "dynamodb", "hbase", "neo4j",

    # ML / AI 
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "reinforcement learning", "transfer learning",
    "feature engineering", "model deployment",
    "time series", "anomaly detection",
    "recommendation systems", "object detection",

    # ML Frameworks
    "tensorflow", "keras", "pytorch", "scikit-learn",
    "xgboost", "lightgbm", "catboost", "hugging face",
    "transformers", "fastai", "spacy", "nltk", "gensim",
    "opencv",

    # Data Libraries
    "pandas", "numpy", "matplotlib", "seaborn",
    "plotly", "scipy", "statsmodels", "dask",

    # BI / Analytics
    "excel", "tableau", "power bi", "looker",
    "google data studio", "qlik", "sas",

    # Cloud Platforms 
    "aws", "azure", "gcp", "google cloud",
    "heroku", "digitalocean", "firebase",

    # MLOps / DevOps
    "docker", "kubernetes", "airflow", "mlflow",
    "kubeflow", "jenkins", "ci/cd", "terraform",
    "ansible",

    # Version Control 
    "git", "github", "gitlab", "bitbucket",

    # Web / APIs 
    "html", "css", "javascript", "react", "nodejs",
    "flask", "fastapi", "django", "rest api",
    "graphql", "typescript", "vue", "angular",

    # Big Data 
    "spark", "hadoop", "hive", "kafka", "flink",
    "databricks", "snowflake", "redshift", "bigquery",

    # Data Science & Analysis 
    "data analysis", "data science", "data engineering",
    "data visualization", "statistical analysis",
    "a/b testing", "hypothesis testing",
    "regression", "classification", "clustering",

    # Soft & Domain Skills 
    "communication", "teamwork", "problem solving",
    "agile", "scrum", "project management",
    "business intelligence", "research",
}

# Skill importance weights (higher = more critical)
SKILL_WEIGHTS = {
    "machine learning": 2.0, "deep learning": 2.0,
    "python": 1.8, "sql": 1.6, "tensorflow": 1.7,
    "pytorch": 1.7, "nlp": 1.8, "data science": 1.7,
    "aws": 1.5, "azure": 1.5, "gcp": 1.5,
    "docker": 1.4, "spark": 1.5, "git": 1.3,
}

# SECTION 2 - TEXT UTILITIES

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Lowercase, remove special chars, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\+\#\. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    clean → tokenise (spaCy) → lemmatise (NLTK) → remove stopwords
    Returns a single string of processed tokens.
    """
    text   = clean_text(text)
    doc    = nlp(text[:1_000_000])          # spaCy token limit guard
    tokens = [
        _lemmatizer.lemmatize(tok.text)
        for tok in doc
        if tok.text not in _stop_words and len(tok.text) > 1
    ]
    return " ".join(tokens)
 
# SECTION 3 - SKILL EXTRACTION  (spaCy + rule-based)

def extract_skills(text: str) -> list[str]:
    """
    Two-pass skill extractor:
      Pass 1 - exact substring match against SKILLS_DB
      Pass 2 - spaCy NER for ORG / PRODUCT entities (bonus)
    Returns a sorted, deduplicated list of skill strings.
    """
    cleaned = clean_text(text)
    found   = set()

    # Pass 1: direct substring matching
    for skill in SKILLS_DB:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned):
            found.add(skill)

    # Pass 2: spaCy named-entity bonus (catches tool names not in DB)
    doc = nlp(cleaned[:500_000])
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT"}:
            candidate = ent.text.lower().strip()
            if candidate in SKILLS_DB:
                found.add(candidate)

    return sorted(found)

# SECTION 4 - FILE READERS

def read_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def get_resume_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext} - use PDF or TXT")

# SECTION 5 - SIMILARITY SCORING  (TF-IDF + Cosine)

def calculate_tfidf_score(resume_text: str, job_text: str) -> float:
    """
    Returns a cosine-similarity score (0–100) between  the preprocessed resume and job description.
    Uses bigrams + unigrams with up to 8 000 features.
    """
    resume_proc = preprocess_text(resume_text)
    job_proc    = preprocess_text(job_text)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8_000,
        sublinear_tf=True,           # dampens extreme term frequencies
    )

    vectors = tfidf.fit_transform([resume_proc, job_proc])
    score   = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(score) * 100, 2)

# SECTION 6 - WEIGHTED SKILL MATCH SCORE

def calculate_skill_score(resume_skills: list, job_skills: list) -> float:
    """
    Computes a weighted skill-overlap score (0–100).
    Skills in SKILL_WEIGHTS contribute more to the final score.
    """
    if not job_skills:
        return 0.0

    total_weight  = 0.0
    matched_weight = 0.0

    for skill in job_skills:
        w = SKILL_WEIGHTS.get(skill, 1.0)
        total_weight += w
        if skill in resume_skills:
            matched_weight += w

    return round((matched_weight / total_weight) * 100, 2) if total_weight else 0.0

# SECTION 7 - COMPOSITE SCORE

def composite_score(tfidf_score: float, skill_score: float) -> float:
     
    return round(0.60 * tfidf_score + 0.40 * skill_score, 2)

# SECTION 8 - ATS DECISION RULES

ATS_TIERS = [
    (85, "⭐ Strong Match",    "Highly recommended - proceed to interview"),
    (70, "✅ Good Match",      "Recommended - strong skill alignment"),
    (55, "🔶 Average Match",   "Consider - partial skill fit, some gaps"),
    (40, "⚠️  Below Average",  "Marginal - significant skill gaps present"),
    ( 0, "❌ Not Suitable",    "Does not meet minimum requirements"),
]

def get_ats_decision(score: float) -> tuple[str, str]:
    """Returns (label, reasoning) based on composite score."""
    for threshold, label, reason in ATS_TIERS:
        if score >= threshold:
            return label, reason
    return "❌ Not Suitable", "Does not meet minimum requirements"


def priority_missing_skills(
    missing_skills: list,
    job_skills: list
) -> dict:
    """
    Classifies missing skills as 'Critical' or 'Recommended'
    based on SKILL_WEIGHTS.
    """
    critical, recommended = [], []
    for skill in missing_skills:
        if SKILL_WEIGHTS.get(skill, 1.0) >= 1.5:
            critical.append(skill)
        else:
            recommended.append(skill)
    return {"Critical": critical, "Recommended": recommended}

# SECTION 9 - CANDIDATE RANKER

def rank_candidates(
    resume_files : list[str],
    job_description: str,
    job_title      : str = "Position"
) -> pd.DataFrame:
    
    job_skills = extract_skills(job_description)
    results    = []

    print(f"\n{'='*62}")
    print(f"  JOB: {job_title}")
    print(f"  Job Skills Detected ({len(job_skills)}): {', '.join(job_skills)}")
    print(f"{'='*62}\n")

    for idx, resume_path in enumerate(resume_files, 1):
        try:
            resume_name = os.path.basename(resume_path)
            print(f"[{idx}/{len(resume_files)}] Processing: {resume_name}")

            # Text extraction 
            resume_text   = get_resume_text(resume_path)
            resume_skills = extract_skills(resume_text)

            # Scoring 
            tfidf_s = calculate_tfidf_score(resume_text, job_description)
            skill_s = calculate_skill_score(resume_skills, job_skills)
            comp_s  = composite_score(tfidf_s, skill_s)

            # ATS decision 
            label, reason = get_ats_decision(comp_s)

            # Skill gap
            missing   = sorted(set(job_skills) - set(resume_skills))
            gap_info  = priority_missing_skills(missing, job_skills)

            results.append({
                "Rank"              : 0,         # filled after sorting
                "Resume"            : resume_name,
                "TF-IDF Score (%)"  : tfidf_s,
                "Skill Score (%)"   : skill_s,
                "Composite Score (%)": comp_s,
                "ATS Decision"      : label,
                "Reasoning"         : reason,
                "Skills Found"      : ", ".join(resume_skills) or "None",
                "Missing Critical"  : ", ".join(gap_info["Critical"]) or "None",
                "Missing Recommended": ", ".join(gap_info["Recommended"]) or "None",
                "_path"             : resume_path,
            })

        except Exception as exc:
            print(f"   ⚠️  Error processing {resume_path}: {exc}")

    # Sort & assign ranks
    df = pd.DataFrame(results)
    if df.empty:
        print("No candidates processed.")
        return df

    df = df.sort_values("Composite Score (%)", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    # Reorder columns for clean output
    col_order = [
        "Rank", "Resume",
        "Composite Score (%)", "TF-IDF Score (%)", "Skill Score (%)",
        "ATS Decision", "Reasoning",
        "Skills Found", "Missing Critical", "Missing Recommended",
    ]
    return df[col_order]

# SECTION 10 - REPORT PRINTER

def print_report(ranking: pd.DataFrame, top_n: int = 3) -> None:

    print("\n" + "="*62)
    print("  CANDIDATE RANKING LEADERBOARD")
    print("="*62)

    leaderboard = ranking[[
        "Rank", "Resume", "Composite Score (%)", "ATS Decision"
    ]]
    print(leaderboard.to_string(index=False))

    print("\n" + "="*62)
    print(f"  TOP {min(top_n, len(ranking))} CANDIDATE DETAIL CARDS")
    print("="*62)

    for _, row in ranking.head(top_n).iterrows():
        print(f"\n{'─'*55}")
        print(f"  #{int(row['Rank'])} {row['Resume']}")
        print(f"{'─'*55}")
        print(f"  Composite Score   : {row['Composite Score (%)']}%")
        print(f"  TF-IDF Score      : {row['TF-IDF Score (%)']}%")
        print(f"  Skill Score       : {row['Skill Score (%)']}%")
        print(f"  ATS Decision      : {row['ATS Decision']}")
        print(f"  Reasoning         : {row['Reasoning']}")
        print(f"\n  ✅ Skills Found:")
        for s in row["Skills Found"].split(", "):
            print(f"     • {s}")
        print(f"\n  🔴 Missing Critical Skills:")
        for s in row["Missing Critical"].split(", "):
            print(f"     • {s}")
        print(f"\n  🟡 Missing Recommended Skills:")
        for s in row["Missing Recommended"].split(", "):
            print(f"     • {s}")

    best = ranking.iloc[0]
    print(f"\n{'='*62}")
    print(f"  🏆 BEST CANDIDATE: {best['Resume']}")
    print(f"     Score: {best['Composite Score (%)']}%  |  {best['ATS Decision']}")
    print("="*62 + "\n")

# SECTION 11 - EXPORT RESULTS

def export_results(ranking: pd.DataFrame, output_csv: str = "screening_results.csv") -> None:
    ranking.to_csv(output_csv, index=False)
    print(f"✅ Results saved to: {output_csv}")

# SECTION 12 - MAIN ENTRY POINT

if __name__ == "__main__":
    CSV_PATH = r"C:/Users/bhara/OneDrive/Desktop/akhil/monster_com-job_sample.csv"

    jobs = pd.read_csv(CSV_PATH, encoding="latin1")
    jobs = jobs[["job_title", "job_description"]].dropna().reset_index(drop=True)

    # Select a job (change index to match the role you want) ─
    JOB_INDEX = 100
    selected_job = jobs.iloc[JOB_INDEX]
    job_title       = selected_job["job_title"]
    job_description = selected_job["job_description"]

    # Add resume file paths here
    # Supports .pdf and .txt files
    RESUME_FILES = [        
    ]

    # Filter to existing files only
    RESUME_FILES = [f for f in RESUME_FILES if os.path.exists(f)]
    if not RESUME_FILES:
        print("\n⚠️  No resume files found - add valid paths to RESUME_FILES list.")
    else:
        # Run the screening pipeline
        ranking = rank_candidates(RESUME_FILES, job_description, job_title)

        # Print the report
        print_report(ranking, top_n=3)

        # Save results to CSV 
        export_results(ranking, output_csv="screening_results.csv")       