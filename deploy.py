
import pickle
from flask import Flask, request, jsonify, app

with open('./model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)



app = Flask('covid19')

@app.route('/predict', methods = ['POST'])

def predict():

    patient = request.get_json()    

    X_dv = dv.transform(patient)
    y_pred = model.predict(X_dv)
    result = {
        'Is covid detected?': y_pred
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug = True, host = '127.0.0.1', port = 5000)
