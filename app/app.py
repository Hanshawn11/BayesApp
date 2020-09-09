import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from flask import Flask, render_template, url_for, request
import joblib

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    cv = CountVectorizer()
    nb = open('nb_spam_model.pkl', 'rb')
    clf = joblib.load(nb)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        ret = clf.predict(vect)
    return render_template('result.html', prediction = ret)
              
if __name__ =='__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)