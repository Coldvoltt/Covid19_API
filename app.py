
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, request, jsonify, app, url_for, render_template


app = Flask(__name__)

model = pickle.load(open('rfModel.pkl', 'rb'))

#@app.route('/')
#def home():
#    return render_template('home.html')


@app.route('/predict', methods = ['POST'])

def predict():

    data = request.json['data']

    df = json.dumps(data)
    df_c = json.loads(df)
    new_data = np.array(list(df_c.values())).reshape(1,-1)

    y_pred = model.predict(new_data)
    Y = ''.join(y_pred)

    result = {'Prediction': Y}
    return jsonify(result)




if __name__ == '__main__':
    app.run(debug = True, host = '127.0.0.1', port = 5000)
