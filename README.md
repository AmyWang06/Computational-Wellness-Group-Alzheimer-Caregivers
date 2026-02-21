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

## Project Structure

ACT_EMA_Analysis/
├── data/ # Interview transcripts & questionnaire data
├── scripts/ # NLP, topic modeling, prediction pipelines
├── outputs/ # Figures, tables, topic visualizations
└── README.md

## Purpose
Transform qualitative interview data into quantitative insights to support research on mental health and caregiving experiences.
