from dataset_to_csv import build_df
from sklearn import feature_extraction, model_selection, pipeline, metrics, svm

df = build_df()

X = df['text']
y = df['party_num_label']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

model = pipeline.Pipeline([
    ('tfidf', feature_extraction.text.TfidfVectorizer()),
    ('clf', svm.LinearSVC())
])

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

accuracy = metrics.accuracy_score(y_predict, y_test)
print(f'SVC accuracy: {accuracy}')
