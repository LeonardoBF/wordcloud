import os
from string import punctuation

import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
import numpy as np

# DEFINITIONS
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
MASK_PATH = os.path.join(BASE_PATH, 'masks')


df = pd.read_csv(os.path.join(DATA_PATH, 'Donald-Tweets!.csv'))
series_tweets = df['Tweet_Text']
text_zaratustra = "Eu vo-lo digo: é preciso ter um caos dentro de si para dar à luz uma estrela cintilante. Eu vo-lo digo: tendes ainda um caos dentro de vós outros. Ai! Aproxima-se o tempo em que o homem já não dará a luz às estrelas; aproxima-se o tempo do mais desprezível dos homens, do que já se não pode desprezar a si mesmo. Olhai! Eu vos mostro o último homem. Que vem a ser isso de amor, de criação, de ardente desejo, de estrela? — pergunta o último homem, revirando os olhos. A terra tornar-se-á então mais pequena, e sobre ela andará aos pulos o último homem, que tudo apouca."

stop_words = set(stopwords.words('english') + list(punctuation) + ['https', 'http'])

mask = np.array(Image.open(os.path.join(MASK_PATH, "cloud2.png")))
wc = WordCloud(background_color="white", mask=mask, max_words=200, stopwords=stop_words)

text = series_tweets.str.cat(sep=' ')
# text = text_zaratustra
wc.generate(text)
wc.to_file(os.path.join(BASE_PATH, "wordcloud.png"))
