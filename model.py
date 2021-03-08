import pickle
import datetime
from sklearn import feature_extraction, model_selection, pipeline, svm, metrics


class Model:
    def __init__(self, filename=None):
        if filename:
            with open(f'./models/{filename}', 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = pipeline.Pipeline([
                ('tfidf', feature_extraction.text.TfidfVectorizer()),
                ('clf', svm.LinearSVC())
            ])

    def train(self, df):
        X = df['text'].values.astype('U')
        y = df['party_num_label']
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

        self.model.fit(X_train, y_train)

        y_predict = self.model.predict(X_test)

        accuracy = metrics.accuracy_score(y_predict, y_test)
        print(f'Trained SVC model accuracy: {accuracy}')

    def predict(self, data):
        return self.model.predict(data)

    def save(self):
        with open(f'./models/model_{datetime.datetime.now()}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
