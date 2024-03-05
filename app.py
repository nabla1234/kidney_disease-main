import os
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

def predict(values, dic):
    if len(values) == 24:
        model = pickle.load(open('models/kidneymodel.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()

            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = float(value)
                except ValueError:
                    return render_template("home.html", message="Please enter valid numerical data")

            to_predict_list = list(map(float, list(to_predict_dict.values())))

            print("Input values:", to_predict_list)
            print("Input values type:", type(to_predict_list))

            pred = predict(to_predict_list, to_predict_dict)

            print("Prediction:", pred)

    except Exception as e:
        print("Error:", str(e))
        message = "Please enter valid data"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
