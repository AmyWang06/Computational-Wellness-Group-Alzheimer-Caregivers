import os
import re
import string
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.models.ldamulticore import LdaMulticore
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
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

# 2. Text cleaning & tokenization
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
    'need'
]
stop_words = set(stopwords.words('english')).union(extra_stopwords)

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

# 3. Split transcripts into question groups
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

# 4. Prepare documents per group
docs_for_modeling = {}
for participant, gdict in group_texts_by_participant.items():
    for gname, raw_text in gdict.items():
        tokens = clean_text_to_tokens(raw_text)
        if tokens:
            docs_for_modeling.setdefault(gname, []).append(tokens)

# 5. Topic modeling (LDA) + pyLDAvis
output_folder = "Results/lda"
os.makedirs(output_folder, exist_ok=True)

num_topics_list = list(range(2, 11))
passes = 10
workers = max(1, os.cpu_count()-1)
random_state = 42

def build_bigrams(all_docs_tokens):
    all_docs_tokens = [doc for doc in all_docs_tokens if isinstance(doc, list) and len(doc) > 0]
    phrases = Phrases(all_docs_tokens, min_count=3, threshold=50)
    phraser = Phraser(phrases)
    bigram_docs = [phraser[doc] for doc in all_docs_tokens]
    return bigram_docs

def train_lda(name, docs_tokens):
    bigram_docs = build_bigrams(docs_tokens)
    id2word = corpora.Dictionary(bigram_docs)
    id2word.filter_extremes(no_below=2, no_above=0.9)
    corpus = [id2word.doc2bow(doc) for doc in bigram_docs]

    best_model = None
    best_coherence = -1
    best_k = None
    for k in num_topics_list:
        model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=k,
                             passes=passes, workers=workers, random_state=random_state)
        cm = CoherenceModel(model=model, texts=bigram_docs, dictionary=id2word, coherence='c_v')
        coherence = cm.get_coherence()
        print(f"[{name}] k={k} coherence={coherence:.4f}")
        if coherence > best_coherence:
            best_coherence = coherence
            best_model = model
            best_k = k

    # Save model and dictionary
    model_path = os.path.join(output_folder, f"lda_{name}_k{best_k}.model")
    best_model.save(model_path)
    with open(os.path.join(output_folder, f"id2word_{name}_k{best_k}.pkl"), 'wb') as f:
        pickle.dump(id2word,f)
    print(f">>> Saved best LDA for '{name}' with k={best_k} (coherence={best_coherence:.4f})")

    # pyLDAvis visualization
    vis_data = gensimvis.prepare(best_model, corpus, id2word)
    vis_path = os.path.join(output_folder, f"lda_vis_{name}_k{best_k}.html")
    pyLDAvis.save_html(vis_data, vis_path)
    print(f">>> Saved pyLDAvis HTML for '{name}' at {vis_path}\n")

    return best_model

if __name__ == "__main__":
    # 6. Exploratory analysis: WordCloud
    all_text_clean = " ".join(df['cleaned_text'])
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=1000
    ).generate(all_text_clean)

    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    wordcloud.to_file("wordcloud.png")

    # Run LDA per group
    for gname, docs_tokens in docs_for_modeling.items():
        train_lda(gname, docs_tokens)