
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from flask import Flask, request, jsonify, app, url_for, render_template

model = pickle.load(open('rfModel.pkl', 'rb'))


app = Flask(__name__)

#@app.route('/')
#def home():
#    return render_template('home.html')


@app.route('/predict', methods = ['POST'])

def predict():

    data = request.json["data"]
    print(data) 

    new_data = data.to_numpy()
    y_pred = model.predict(new_data)
    result = {
        'Is covid detected?': y_pred
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True, host = '127.0.0.1', port = 5000)
