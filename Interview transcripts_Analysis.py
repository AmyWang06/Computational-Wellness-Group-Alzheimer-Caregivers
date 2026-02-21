import zipfile
import os
import re
import pickle
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


###4. LDA Model(Topic Modelling)
# -----------------------
# Step 1: Load the data
# -----------------------
# Unzip the file
with zipfile.ZipFile("./data/NIPS Papers.zip", "r") as zip_ref:
    zip_ref.extractall("temp")

# Read the CSV file
papers = pd.read_csv("temp/NIPS Papers/papers.csv")
print("First 5 rows of the dataset:")
print(papers.head())

# -----------------------
# Step 2: Data cleaning
# -----------------------
# Keep only the paper text column, randomly select 100 papers
papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)
print("After selecting 100 papers:")
print(papers.head())

# Remove punctuation and convert to lowercase
papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())
print("Processed text:")
print(papers['paper_text_processed'].head())

# -----------------------
# Step 3: Exploratory analysis (WordCloud)
# -----------------------
long_string = ' '.join(list(papers['paper_text_processed'].values))
wordcloud = WordCloud(
    background_color="white",
    max_words=1000,
    contour_width=3,
    contour_color='steelblue'
)
wordcloud.generate(long_string)
wordcloud.to_image().show()  # Display the WordCloud

# -----------------------
# Step 4: Prepare input for LDA
# -----------------------
# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Tokenization and cleaning
def sent_to_words(sentences):
    for sentence in sentences:
        yield simple_preprocess(str(sentence), deacc=True)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)
print("First 30 tokens of the first document:")
print(data_words[0][:30])

# Create dictionary and corpus
id2word = corpora.Dictionary(data_words)
texts = data_words
corpus = [id2word.doc2bow(text) for text in texts]
print("First 30 BOW entries of the first document:")
print(corpus[0][:30])

# -----------------------
# Step 5: Train the LDA model
# -----------------------
num_topics = 10
lda_model = gensim.models.LdaMulticore(
    corpus=corpus,
    id2word=id2word,
    num_topics=num_topics
)

# Print the 10 topics with keywords
print("LDA Topics:")
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# -----------------------
# Step 6: Analyze and visualize the LDA model results (pyLDAvis)
# -----------------------
pyLDAvis.enable_notebook()

# Create results folder if not exists
os.makedirs('./results', exist_ok=True)
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))

# Prepare and save visualization data
LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

# Load prepared data (optional)
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

# Save visualization as HTML
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
print("LDAvis HTML saved at './results/ldavis_prepared_10.html'")