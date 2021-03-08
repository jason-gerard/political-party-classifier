import glob
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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

    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_words = []

    for word in words:
        # remove punctuation
        word = word.translate(table)

        # remove non alphabetic tokens and stop words
        if word.isalpha() and word not in stop_words:
            # remove word stems with lemmatization
            word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            clean_words.append(word)

    return ' '.join(clean_words)


def build_df():
    return pd.read_csv('./raw_dataset/training_set.csv')


def dataset_to_csv():
    training_set_dir = './raw_dataset/training_set'
    training_set_file_names = [file for file in glob.glob(f'{training_set_dir}/*.txt')]

    raw_data = []

    for index, filename in enumerate(training_set_file_names):
        row = []
        with open(filename) as f:
            row.append(f.read())
        party_indicator = filename.split('_')[-1][0]
        row.append(party_indicator)
        # 0 -> dem
        # 1 -> repub
        party_num_label = 0 if party_indicator == 'D' else 1
        row.append(party_num_label)

        raw_data.append(row)

        if index % 10 == 0:
            print(f'Processed {index} files...')

    df = pd.DataFrame(raw_data)
    df.rename(columns={0: 'text', 1: 'party', 2: 'party_num_label'}, inplace=True)

    print(f'Cleaning {df.shape[0]} rows of text')
    df['text'] = df['text'].apply(clean_text)

    df.to_csv('./raw_dataset/training_set.csv', index=False)


if __name__ == '__main__':
    print('Building csv...')
    dataset_to_csv()
