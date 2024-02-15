import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from textblob import Word

df = pd.read_csv("amazon_reviews.csv")

df['reviewText'] = df['reviewText'].str.lower()
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')
df['reviewText'] = df['reviewText'].str.replace('\d', '')

#import nltk
#nltk.download('stopwords')

sw = stopwords.words('english')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

drops = temp_df[temp_df <= 2]

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

#nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


#Sentiment Analysis
#nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

liste =  []
liste2 =  []
for i in range(0,4915):
    liste.append(sia.polarity_scores(df["reviewText"][i])["compound"])
    sentiment = sia.polarity_scores(df["reviewText"][i])["compound"]
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
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(df["reviewText"])


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(df["reviewText"])

log_model = LogisticRegression().fit(X_tf_idf_word, df["overall"])

random_review = ["thats great idea very usefull i recommended it is"]

new_review = TfidfVectorizer().fit(df["reviewText"]).transform(random_review)

print("Ornek overall tahminlemesi(LogisticRegression): ",log_model.predict(new_review))










