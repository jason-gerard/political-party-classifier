import glob
import pandas as pd
import string
import nltk
import psycopg2
import psycopg2.extras
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import db


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


def convote_database_to_df():
    table_name = 'convote_dataset'

    conn = db.make_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(f'''
        SELECT
            text,
            party,
            party_num_label
        FROM {table_name}
    ''')

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    df = pd.DataFrame(rows, columns=['text', 'party', 'party_num_label'])

    return df


def convote_dataset_to_db():
    table_name = 'convote_dataset'
    training_set_dir = '../raw_dataset/training_set'
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

    conn = db.make_conn()

    rows = [tuple(x) for x in df.to_numpy()]
    column_names = ','.join(list(df.columns))

    query = f'INSERT INTO {table_name} ({column_names}) VALUES(%s,%s,%s)'
    cursor = conn.cursor()
    try:
        psycopg2.extras.execute_batch(cursor, query, rows)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(f'Error: {error}')
        conn.rollback()
        cursor.close()
        return

    print("execute_batch() done")
    cursor.close()


if __name__ == '__main__':
    convote_dataset_to_db()
