import os
from string import punctuation

import pandas as pd
from nltk.corpus import stopwords

# DEFINITIONS
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')


df = pd.read_csv(os.path.join(DATA_PATH, 'Donald-Tweets!.csv'))
series_tweets = df['Tweet_Text']

stop_words = set(stopwords.words('portuguese') + list(punctuation))
