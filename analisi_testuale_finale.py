# 📦 Installazione librerie (eseguire solo in Google Colab)
# Commenta questa cella se lavori in locale con librerie già installate
# !pip install spacy nltk wordcloud matplotlib pandas scikit-learn
# !python -m spacy download it_core_news_sm

# 📚 Importa le librerie
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('stopwords')
nlp = spacy.load('it_core_news_sm')

# 📄 Carica e pulisci la trascrizione
with open("trascrizione.txt", "r", encoding="utf-8") as f:
    testo = f.read().lower()

stop_words = set(stopwords.words('italian'))
doc = nlp(testo)
tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]

# 🔢 Frequenza parole
frequenze = Counter(tokens)
df_freq = pd.DataFrame(frequenze.most_common(20), columns=['Parola', 'Frequenza'])
df_freq.plot(kind='bar', x='Parola', y='Frequenza', legend=False, figsize=(10,5))
plt.title('🔠 Parole più frequenti')
plt.show()

# ☁️ Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("☁️ Word Cloud")
plt.show()

# 🧩 Topic Modeling (LDA)
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=stop_words)
X = vectorizer.fit_transform([' '.join(tokens)])
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

for idx, topic in enumerate(lda.components_):
    print(f"\n🧵 Topic {idx + 1}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])