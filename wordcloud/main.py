import os
from string import punctuation

import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from settings import *


def define_stopwords(language='english', adicional_stopwords=[]):
    return set(stopwords.words(language) + list(punctuation) + adicional_stopwords)


def define_wordcloud(stopwords, maskname='cloud.png'):
    mask = np.array(Image.open(os.path.join(MASK_PATH, maskname)))
    wc = WordCloud(background_color="white", mask=mask, max_words=300, stopwords=stopwords,
                   relative_scaling=0.15, collocations=False)
    return wc


def generate_wordcloud(wc, visualize=True, save_to_file=False, filename='wordcloud.png'):
    if visualize:
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.figure()

    if save_to_file:
        wc.to_file(os.path.join(WORDCLOUD_PATH, filename))



df = pd.read_csv(os.path.join(DATA_PATH, 'Donald-Tweets!.csv'))
series_tweets = df['Tweet_Text']
text = series_tweets.str.cat(sep=' ').lower()

adicional_stopwords = ['http', 'https']
stopwords = define_stopwords(adicional_stopwords=adicional_stopwords)

wc = define_wordcloud(stopwords)
wc.generate(text)

generate_wordcloud(wc, visualize=True, save_to_file=False, filename='WC-DonaldTweets.png')
