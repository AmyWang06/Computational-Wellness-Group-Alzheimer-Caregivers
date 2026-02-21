# ACT EMA Interview Analysis
This project analyzes interview transcripts and questionnaire data from the ACT EMA study to uncover patterns related to **stress**, **loneliness**, **caregiving burden**, and **coping strategies** using NLP and machine learning.

## Key Features
- Text preprocessing and segmentation by interview questions  
- Exploratory NLP: **word frequency**, **TF-IDF**, **n-grams**, **sentiment analysis**  
- Topic modeling (**LDA**) with coherence-based topic selection  
- Semantic embeddings using **Sentence-BERT**  
- Predictive modeling of mental health outcomes (classification & regression)  
- Demographic and questionnaire-based statistical analysis  
- Visualization (word clouds, topic maps, correlation heatmaps)

## Methods & Tools
- Python, pandas, scikit-learn, gensim  
- TF-IDF, LDA, Sentence-BERT embeddings  
- Cross-validation, **AUC / R²** evaluation  
- pyLDAvis, matplotlib, seaborn

## Scripts Overview
**Alzheimer_Interview_Analysis.py** 

Preprocesses and analyzes caregiver interview transcripts using NLP techniques, including text cleaning, TF-IDF, n-grams, sentiment analysis, and exploratory visualizations.

**Alzheimer_Interview_Topic Modeling.py**

Performs LDA topic modeling on interview transcripts with coherence-based topic selection and interactive visualization using pyLDAvis.

**Alzheimer_Interview_Prediction Score.py**

Builds predictive models to estimate mental health outcomes (e.g., stress, loneliness, caregiver burden) from interview text using TF-IDF features and Sentence-BERT embeddings.

**Caregiver_Demographics.py**

Analyzes caregiver demographic variables and reports descriptive statistics and subgroup comparisons.

**Caregiver_Survey Analysis.py**

Processes questionnaire data to compute scale scores, apply clinical cutoffs, and examine correlations among mental health measures.

## Purpose
Transform qualitative interview data into quantitative insights to support research on mental health and caregiving experiences.
