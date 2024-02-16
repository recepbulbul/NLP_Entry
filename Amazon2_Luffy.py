import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import Word

df = pd.read_excel("amazon.xlsx")

df['Review'] = df['Review'].str.lower()

df['Review'] = df['Review'].str.replace('[^\w\s]', '')

df['Review'] = df['Review'].str.replace('\d', '')

#import nltk
# nltk.download('stopwords')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

#nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    
text = " ".join(i for i in df.Review)

luffy_mask = np.array(Image.open("Luffy.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=luffy_mask,
               contour_width=3,
               contour_color="darkblue")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()























