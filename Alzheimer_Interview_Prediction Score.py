import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
import string
import pyreadstat


# Preprocessing
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def load_transcripts(folder_path):
    """
    Load all .txt interview transcripts from the folder.
    Each file corresponds to one participant.
    """
    rows = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            participant_id = os.path.splitext(filename)[0]

            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()

            rows.append({
                "participant_id": participant_id,
                "raw_text": text
            })

    return pd.DataFrame(rows)

def extract_answers_only(text):
    """
    Extract only participant responses from sections R1.x to R4.x.
    Ignore all questions (Q*) and also ignore R5.x and R6.x.
    """
    answers = []

    # Find all response blocks, like [R1.1] some text
    pattern = r"\[R(\d)\.(\d+)\](.*?)(?=\[R|\[Q|\Z)"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for section_num, item_num, response in matches:
        section_num = int(section_num)

        # Keep only R1.*, R2.*, R3.*, R4.*
        if 1 <= section_num <= 4:
            cleaned = response.strip()
            answers.append(cleaned)

    return "\n".join(answers)

def clean_text(text):
    """
    Lowercase text, remove punctuation, and remove English stopwords.
    This prepares text for NLP modeling.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

def preprocess_transcripts(df):
    """
    Apply extraction and cleaning to the entire dataset.
    """
    df["answers_1_4"] = df["raw_text"].apply(extract_answers_only)
    df["clean_text"] = df["answers_1_4"].apply(clean_text)
    return df

def normalize_study_id(x):
    """
    Standardize participant identifiers by extracting the numeric
    component of the StudyID and converting it to a consistent
    'ACTXXXX' format to ensure proper alignment across data sources.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    digits = re.findall(r"\d+", x)
    if len(digits) == 0:
        return np.nan

    num = digits[0].zfill(4)
    return f"ACT{num}"

# Classic NLP
def build_tfidf_features(df, max_features=5000):
    """
    Create TF-IDF features for classical machine learning models.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["clean_text"])

    return X, vectorizer

# LLM
def build_embeddings(df):
    """
    Build semantic embeddings using a pre-trained Sentence-BERT model.
    Each participant's text becomes a 768-dimensional vector.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(df["clean_text"].tolist(), show_progress_bar=True)

    return np.array(embeddings)

# Run

# 1. Load all interview transcripts
df = load_transcripts("data/Segmented and Tagged Transcripts")
print(f"Loaded {len(df)} transcripts.")

# 2. Preprocess transcripts: extract answers (R1-R4) + clean text
df = preprocess_transcripts(df)
df = df.rename(columns={"participant_id": "StudyID"})
df["StudyID"] = df["StudyID"].apply(normalize_study_id)
df = df.reset_index(drop=True)
id_to_idx = pd.Series(df.index, index=df["StudyID"])
print("Preprocessing completed. Sample cleaned text:")
print(df["clean_text"].head(3))

# 3. Save cleaned text for inspection
df.to_csv("processed_transcripts.csv", index=False)
print("Cleaned transcripts saved to 'processed_transcripts.csv'.")

# 4. Build classical NLP features (TF-IDF)
X_tfidf, vectorizer = build_tfidf_features(df)
print("TF-IDF feature matrix shape:", X_tfidf.shape)

# 5. Build LLM embeddings (Sentence-BERT)
X_llm = build_embeddings(df)
print("LLM embedding matrix shape:", X_llm.shape)

print("✔ Text preprocessing and feature construction completed.")
print("Next step: merge with survey scores and train prediction models.")

# 6. Load Visit 2 questionnaire data (.sav)
sav_path = "ACT V1 V2 Questionnaires_Cytokines_ACT EMA Participants (1).sav"
survey_df, meta = pyreadstat.read_sav(sav_path)

survey_df = survey_df[[
    "StudyID",
    "UCLALS_Total_V2",
    "PSS_Total_V2",
    "ZBI_Total12_V2",
    "ZBI_Total13_V2",
    "ZBI_CutOff_V2"
]]
survey_df["StudyID"] = survey_df["StudyID"].apply(normalize_study_id)

# Binary variables
survey_df["UCLALS_bin"] = (survey_df["UCLALS_Total_V2"] >= 34).astype(int)
survey_df["PSS_bin"] = (survey_df["PSS_Total_V2"] >= 13).astype(int)
survey_df["ZBI_bin"] = (survey_df["ZBI_Total12_V2"] >= 20).astype(int)

print("Visit 2 questionnaire data loaded.")

# 7. Load coping scores (Visit 2)
coping_df = pd.read_csv("ACT - Visit 2 EMA Participant Questionnaires_with COPE Scores.csv")

coping_df = coping_df[[
    "StudyID",
    "Problem_Focused_Coping",
    "Emotion_Focused_Coping",
    "Avoidant_Coping"
]]
coping_df["StudyID"] = coping_df["StudyID"].apply(normalize_study_id)

print("✔ Coping scores loaded.")

# 8. Merge interview text with survey & coping data
df_merged = (
    df
    .merge(survey_df, on="StudyID", how="inner")
    .merge(coping_df, on="StudyID", how="inner")
)

aligned_idx = id_to_idx.loc[df_merged["StudyID"]].values

X_tfidf_aligned = X_tfidf[aligned_idx]
X_llm_aligned = X_llm[aligned_idx]
y = df_merged["UCLALS_bin"].values

print("✔ Final merged dataset shape:", df_merged.shape)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge

def run_binary_classification(X, y, label):
    model = LogisticRegression(max_iter=1000)
    auc = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc"
    )
    print(f"{label}: AUC = {auc.mean():.3f} ± {auc.std():.3f}")

def run_regression(X, y, label):
    model = Ridge(alpha=1.0)
    r2 = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )
    print(f"{label}: R² = {r2.mean():.3f} ± {r2.std():.3f}")

# 9. TF-IDF models
# Binary outcomes
run_binary_classification(X_tfidf_aligned, df_merged["UCLALS_bin"], "TF-IDF UCLA Loneliness")
run_binary_classification(X_tfidf_aligned, df_merged["PSS_bin"], "TF-IDF PSS")
run_binary_classification(X_tfidf_aligned, df_merged["ZBI_bin"], "TF-IDF ZBI")
mask = df_merged["ZBI_CutOff_V2"].notna()
run_binary_classification(
    X_tfidf_aligned[mask],
    df_merged.loc[mask, "ZBI_CutOff_V2"],
    "TF-IDF ZBI Cutoff"
)

# Continuous outcomes
def run_regression_safe(X, y, label):
    mask = y.notna()
    run_regression(X[mask], y.loc[mask], label)

run_regression_safe(X_tfidf_aligned, df_merged["UCLALS_Total_V2"], "TF-IDF UCLA Loneliness")
run_regression_safe(X_tfidf_aligned, df_merged["PSS_Total_V2"], "TF-IDF PSS")
run_regression_safe(X_tfidf_aligned, df_merged["ZBI_Total12_V2"], "TF-IDF ZBI (12-item)")
run_regression_safe(X_tfidf_aligned, df_merged["Problem_Focused_Coping"], "TF-IDF Problem-focused coping")
run_regression_safe(X_tfidf_aligned, df_merged["Emotion_Focused_Coping"], "TF-IDF Emotion-focused coping")
run_regression_safe(X_tfidf_aligned, df_merged["Avoidant_Coping"], "TF-IDF Avoidant coping")

# 10. LLM embedding models
run_binary_classification(X_llm_aligned, df_merged["UCLALS_bin"], "LLM UCLA Loneliness")
run_binary_classification(X_llm_aligned, df_merged["PSS_bin"], "LLM PSS")
run_binary_classification(X_llm_aligned, df_merged["ZBI_bin"], "LLM ZBI")

run_regression_safe(X_llm_aligned, df_merged["UCLALS_Total_V2"], "LLM UCLA Loneliness")
run_regression_safe(X_llm_aligned, df_merged["PSS_Total_V2"], "LLM PSS")
run_regression_safe(X_llm_aligned, df_merged["ZBI_Total12_V2"], "LLM ZBI (12-item)")
run_regression_safe(X_llm_aligned, df_merged["Problem_Focused_Coping"], "LLM Problem-focused coping")
run_regression_safe(X_llm_aligned, df_merged["Emotion_Focused_Coping"], "LLM Emotion-focused coping")
run_regression_safe(X_llm_aligned, df_merged["Avoidant_Coping"], "LLM Avoidant coping")