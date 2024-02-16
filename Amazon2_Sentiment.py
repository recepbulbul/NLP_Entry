import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from textblob import Word
from sklearn.model_selection import cross_val_score

df = pd.read_excel("amazon.xlsx")

df['Review'] = df['Review'].str.lower()
df['Review'] = df['Review'].str.replace('[^\w\s]', '')
df['Review'] = df['Review'].str.replace('\d', '')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()

drops = temp_df[temp_df <= 2]

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

#nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

sia = SentimentIntensityAnalyzer()

liste =  []
liste2 =  []
for i in range(0,5611):
    liste.append(sia.polarity_scores(df["Review"][i])["compound"])
    sentiment = sia.polarity_scores(df["Review"][i])["compound"]
    if sentiment > 0:
        liste2.append("pos")
    elif sentiment == 0:
        liste2.append("notr")
    else:
        liste2.append("neg")
df["Polarity_Score"] = liste
df["Sentiment"] = liste2

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 5))
df['Polarity_Score'] = scaler.fit_transform(df['Polarity_Score'].values.reshape(-1, 1))


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(df["Review"])


log_model = LogisticRegression().fit(X_tf_idf_word, df["Sentiment"])

print("\nKurulan Modelin Doğruluk Değeri",cross_val_score(log_model,
                X_tf_idf_word,
                df["Sentiment"],
                scoring="accuracy",
                cv=10).mean()*100)































