import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = ['Campaign Product Name', 'Region', 'Amount Spent (INR)'])                         
    
    prediction = model.predict(data_unseen)

    output = prediction


    return render_template('index.html', prediction_text='Result is')

if __name__ == "__main__":
    app.run(debug=True)