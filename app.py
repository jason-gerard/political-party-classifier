import numpy as np
from flask import Flask, request

from model import Model
from utils import db

app = Flask(__name__)
model = Model(filename='model_2021-03-07 18:17:03.523405.pkl')


@app.route('/predict', methods=['GET'])
def predict():
    text = request.form.get('text')
    prediction = model.predict(np.array([text]))[0]

    table_name = 'convote_predictions'

    conn = db.make_conn()
    cursor = conn.cursor()
    cursor.execute(f'''
        INSERT
        INTO {table_name} (text, party, party_num_label)
        VALUES ('{text}', '{'D' if prediction == 0 else 'R'}', {prediction})
    ''')

    conn.commit()
    cursor.close()

    return 'Democrat' if prediction == 0 else 'Republican'


if __name__ == '__main__':
    app.run(debug=True)
