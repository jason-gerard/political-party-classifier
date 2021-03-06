import string
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('./raw_dataset/training_set.csv')


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    text = text.lower()
    words = word_tokenize(text)

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in words]

    # remove non alphabetic tokens
    words = [word for word in words if word.isalpha()]

    # remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # remove word stems with lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]

    return ' '.join(words)


df2 = df.copy()[:10]
print(df2.head())
df2['text'] = df2['text'].apply(clean_text)
print(df2.head())
