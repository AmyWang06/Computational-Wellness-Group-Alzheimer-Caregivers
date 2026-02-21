import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import OrderedDict

# 1. Load data
data_folder = "data/Segmented and Tagged Transcripts/"
txt_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.txt')]
rows = []
for fname in txt_files:
    path = os.path.join(data_folder, fname)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()
    participant = os.path.splitext(fname)[0]
    rows.append({'participant': participant, 'Transcription': text})
df = pd.DataFrame(rows)


# 2. Text cleaning & preprocessing
extra_stopwords = [
    'like','know','just','um','yeah','kind','going','think','don',
    'did','say','uh','maybe','doing','help','thing','probably',
    'make','mean','would','im','get','dont','thats','go',
    'q1','q2','q3','q4','q5','q6','r1','r2','r3','r4','r5','r6',
    'one','much','lot','right','want','something','little','every',
    'well','didnt','hes','cant','oh','yes','see','sure','could',
    'sometimes','might','still','come','getting','makes',
    'okay','somewhat','day','days','time','people','home','use',
    'take','change','care','things','youre','said','bit','trying',
    '100','hours','times','two','stuff','able','example','point','per',
    'also','got', 'really', 'extremely', 'slightly', 'anything', 'actually', 'find',
    'moderately', 'everything', 'completely', 'guess', 'many', 'pretty', 'feel',
    'need', 'around','even'
]
stop_words = list(set(stopwords.words('english')).union(extra_stopwords))

def clean_text_to_tokens(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

def clean_text_to_string(text):
    tokens = clean_text_to_tokens(text)
    return " ".join(tokens)

df['cleaned_tokens'] = df['Transcription'].apply(clean_text_to_tokens)
df['cleaned_text'] = df['Transcription'].apply(clean_text_to_string)


# 3. Attempt to split transcripts into question groups
question_groups = {
    "Caregiving": ["1.", "2.", "3."],
    "Stress_Coping": ["4."],
    "Tech_Tool_Experience": ["5."],
    "Counterfactual_tool": ["6."]
}

normalized_markers = {}
for gname, markers in question_groups.items():
    normalized_markers[gname] = [m.lower().strip() for m in markers]

def extract_responses_only(text):
    response_pattern = r'\[R\d+\.\d+\](.*?)((?=\[R\d+\.\d+\])|$)'
    matches = re.findall(response_pattern, text, flags=re.DOTALL)
    responses = [m[0].strip() for m in matches]
    return " ".join(responses).replace('\n', ' ').strip()

def split_transcript_by_markers(transcript, normalized_markers):
    text = transcript.replace('\n', ' ')
    text_lower = text.lower()

    result = OrderedDict()
    for g in ["Caregiving", "Stress_Coping", "Tech_Tool_Experience", "Counterfactual_tool"]:
        result[g] = ""

    hits = []
    for gname, prefixes in normalized_markers.items():
        for prefix in prefixes:
            for match in re.finditer(re.escape(prefix), text_lower):
                hits.append((match.start(), gname))

    if not hits:
        result['All'] = transcript
        return result

    hits_sorted = sorted(hits, key=lambda x: x[0])

    for i, (start_pos, gname) in enumerate(hits_sorted):
        end_pos = hits_sorted[i + 1][0] if i + 1 < len(hits_sorted) else len(text)
        snippet = text[start_pos:end_pos].strip()
        result[gname] += " " + snippet if result[gname] else snippet

    result['All'] = extract_responses_only(transcript)
    return result


group_texts_by_participant = {}
for idx, row in df.iterrows():
    participant = row['participant']
    transcript = row['Transcription']
    group_texts = split_transcript_by_markers(transcript, normalized_markers)
    group_texts_by_participant[participant] = group_texts


# 4. Analysis functions
def word_frequency_from_text(text, top_n=20):
    tokens = clean_text_to_tokens(text)
    c = Counter(tokens)
    return pd.DataFrame(c.most_common(top_n), columns=['Word', 'Frequency'])

def sentiment_sentences_from_text(text):
    sents = sent_tokenize(text)
    rows = []
    for s in sents:
        if s.strip():
            tb = TextBlob(s)
            rows.append({'sentence': s, 'polarity': tb.sentiment.polarity, 'subjectivity': tb.sentiment.subjectivity})
    if rows:
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame(columns=['sentence','polarity','subjectivity'])

def extract_tfidf_keywords(corpus_texts, top_n=10):
    if not corpus_texts or all(not str(t).strip() for t in corpus_texts):
        return pd.DataFrame(columns=['keyword', 'tfidf'])

    n_docs = len(corpus_texts)
    vectorizer = TfidfVectorizer(
        stop_words = stop_words,
        max_df=0.9,
        min_df=1
    )

    X = vectorizer.fit_transform(corpus_texts)
    names = vectorizer.get_feature_names_out()
    sums = np.asarray(X.sum(axis=0)).ravel()

    dfk = pd.DataFrame({'keyword': names, 'tfidf': sums})
    dfk = dfk.sort_values('tfidf', ascending=False).head(top_n).reset_index(drop=True)

    return dfk

def cooccurrence_matrix_from_texts(texts, top_n=20):
    if not texts:
        return pd.DataFrame()
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    M = (X.T * X).toarray()
    np.fill_diagonal(M, 0)
    dfm = pd.DataFrame(M, index=terms, columns=terms)
    top_terms = dfm.sum(axis=1).nlargest(top_n).index
    return dfm.loc[top_terms, top_terms]

def ngram_counts(texts, n=2, top_n=10):
    if not texts:
        return pd.DataFrame(columns=['ngram','count'])
    vectorizer = CountVectorizer(ngram_range=(n,n), stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    names = vectorizer.get_feature_names_out()
    dfng = pd.DataFrame({'ngram': names, 'count': sums})
    dfng = dfng.sort_values('count', ascending=False).head(top_n).reset_index(drop=True)
    return dfng


# 5. Run analyses per group (aggregated across participants)
# Prepare aggregated texts per group
groups_all_texts = {}
for participant, gdict in group_texts_by_participant.items():
    for gname, text in gdict.items():
        groups_all_texts.setdefault(gname, []).append(text)

# For each group, compute:
# - Word frequency (top 20)
# - Sentiment by sentences (all sentences aggregated)
# - TF-IDF top keywords
# - Co-occurrence matrix (top 20)
# - Bi-grams and Tri-grams top 10
analysis_results = {}

for gname, texts in groups_all_texts.items():
    combined_text = " ".join(texts).strip()
    # Word frequency
    wf = word_frequency_from_text(combined_text, top_n=20)
    # Sentiment
    sentiment_df = pd.concat([sentiment_sentences_from_text(t) for t in texts], ignore_index=True)
    # TF-IDF
    tfidf = extract_tfidf_keywords(texts, top_n=20)
    # Co-occurrence
    cooc = cooccurrence_matrix_from_texts(texts, top_n=20)
    # N-grams
    bigrams = ngram_counts(texts, n=2, top_n=20)
    trigrams = ngram_counts(texts, n=3, top_n=20)
    # Keyword-level sentiment (average polarity/subjectivity where keyword appears in sentence)
    keyword_sentiments = {}
    keywords_for_sentiment = tfidf['keyword'].tolist() if not tfidf.empty else []
    for kw in keywords_for_sentiment:
        mask_sentences = []
        for t in texts:
            for s in sent_tokenize(t):
                kw_tokens = kw.split()
                pattern = r'\b' + r'\s+'.join(map(re.escape, kw_tokens)) + r'\b'
                if re.search(pattern, s, flags=re.I):
                    mask_sentences.append(s)
        if mask_sentences:
            df_s = pd.DataFrame([TextBlob(s).sentiment for s in mask_sentences], columns=['polarity','subjectivity'])
            keyword_sentiments[kw] = {'avg_polarity': df_s['polarity'].mean(), 'avg_subjectivity': df_s['subjectivity'].mean()}
        else:
            keyword_sentiments[kw] = {'avg_polarity': np.nan, 'avg_subjectivity': np.nan}
    analysis_results[gname] = {
        'combined_text': combined_text,
        'word_freq': wf,
        'sentences_sentiment': sentiment_df,
        'tfidf': tfidf,
        'cooccurrence': cooc,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'keyword_sentiments': pd.DataFrame.from_dict(keyword_sentiments, orient='index')
    }


# 6. Save results to files
output_folder = "Results"
os.makedirs(output_folder, exist_ok=True)

# Save aggregated per-group CSVs
for gname, res in analysis_results.items():
    safe_name = re.sub(r'[^0-9a-zA-Z_\-]+', '_', gname)
    # Word freq
    wf_path = os.path.join(output_folder, f"{safe_name}_word_freq.csv")
    res['word_freq'].to_csv(wf_path, index=False)
    # Sentiment sentences
    s_path = os.path.join(output_folder, f"{safe_name}_sentences_sentiment.csv")
    res['sentences_sentiment'].to_csv(s_path, index=False)
    # TF-IDF
    tfidf_path = os.path.join(output_folder, f"{safe_name}_tfidf.csv")
    res['tfidf'].to_csv(tfidf_path, index=False)
    # Keyword sentiments
    ks_path = os.path.join(output_folder, f"{safe_name}_keyword_sentiments.csv")
    res['keyword_sentiments'].to_csv(ks_path, index=True)
    # Bigrams / Trigrams
    res['bigrams'].to_csv(os.path.join(output_folder, f"{safe_name}_bigrams.csv"), index=False)
    res['trigrams'].to_csv(os.path.join(output_folder, f"{safe_name}_trigrams.csv"), index=False)
    # Co-occurrence (as csv)
    if not res['cooccurrence'].empty:
        res['cooccurrence'].to_csv(os.path.join(output_folder, f"{safe_name}_cooccurrence.csv"))

# Save per-participant raw extracted groups (sorted)
group_rows = []
for participant, gdict in group_texts_by_participant.items():
    for gname, text in gdict.items():
        group_rows.append({'participant': participant, 'group': gname, 'text': text})

# Sort first by group name, then by participant
group_rows_sorted = sorted(group_rows, key=lambda x: (x['group'], x['participant']))
pd.DataFrame(group_rows_sorted).to_csv(os.path.join(output_folder, "per_participant_group_texts.csv"), index=False)

# 8. Summary CSV tying groups to top keywords and mean polarity
summary_rows = []
for gname, res in analysis_results.items():
    top_keywords = res['tfidf']['keyword'].head(10).tolist() if not res['tfidf'].empty else []
    mean_polarity = res['sentences_sentiment']['polarity'].mean() if not res['sentences_sentiment'].empty else np.nan
    summary_rows.append({'group': gname, 'top_keywords': ";".join(top_keywords), 'mean_polarity': mean_polarity})
pd.DataFrame(summary_rows).to_csv(os.path.join(output_folder, "groups_summary.csv"), index=False)


# 9. Print paths of saved outputs
print("Analysis complete. Outputs saved to folder:", output_folder)
