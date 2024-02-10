import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv", sep= "\t", quoting= 3)

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 


for i in range(0,1000):
    cleantext = re.sub("[^a-z A-Z]", "", dataset["Review"][i])
    cleantext = cleantext.lower()
    cleantext = cleantext.split()
    dataset["Review"][i] = cleantext
   
stops = stopwords.words("english")
stops.remove("not")

for i in range(0,1000):
    liste = []
    for kelime in dataset["Review"][i]:
        if kelime in stops:
            continue
        else:
            liste.append(kelime)
    dataset["Review"][i] = liste

ps = PorterStemmer()

for i in range(0,1000):
    liste = []
    for kelime in dataset["Review"][i]:
        liste.append(ps.stem(kelime))
    dataset["Review"][i] = liste


for i in range(0,1000):
    cumle = " ".join(dataset["Review"][i])
    dataset["Review"][i] = cumle
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1000)

X = cv.fit_transform(dataset["Review"]).toarray()
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                          random_state=1)

from sklearn.svm import SVC

classifier = SVC(probability=True, kernel="rbf")

classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
AccurS = accuracy_score(Y_test, Y_pred)
















