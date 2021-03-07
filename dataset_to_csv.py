import glob
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def build_raw_df():
    training_set_dir = './raw_dataset/training_set'
    training_set_file_names = [file for file in glob.glob(f'{training_set_dir}/*.txt')]

    raw_data = []

    for filename in training_set_file_names:
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

    df = pd.DataFrame(raw_data)
    df.rename(columns={0: 'text', 1: 'party', 2: 'party_num_label'}, inplace=True)

    return df


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


def build_df():
    return pd.read_csv('./raw_dataset/training_set.csv')


def dataset_to_csv():
    df = build_raw_df()
    df['text'] = df['text'].apply(clean_text)

    df.to_csv('./raw_dataset/training_set.csv', index=False)


if __name__ == '__main__':
    print('Building csv...')
    dataset_to_csv()
