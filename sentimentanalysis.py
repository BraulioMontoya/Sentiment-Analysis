import textpreprocessing

import matplotlib.pyplot as plt
import pandas as pd

from textblob import TextBlob
from wordcloud import WordCloud

def get_polarity(text):
    analysis = TextBlob(text)
    
    return analysis.polarity

def get_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    
    if polarity == 0:
        return 'neutral'
    
    return 'negative'

df = pd.read_csv('dataset.csv')
df['clean_text'] = df['text'].apply(textpreprocessing.clean_text)
df['lemmatize'] = df['clean_text'].apply(textpreprocessing.lemmatize)
df['polarity'] = df['clean_text'].apply(get_polarity)
df['sentiment'] = df['polarity'].apply(get_sentiment)

plt.pie(df['sentiment'].value_counts(), labels = ['neutral', 'positive', 'negative'], autopct = '%1.1f%%')
plt.title('tweet sentiment analysis')
plt.show()

text = ' '.join(df['lemmatize'])

wordcloud = WordCloud().generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()