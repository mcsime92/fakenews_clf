from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
import re
import os

app = Flask(__name__)


# Used by bow pickle file
def clean_article(article):
    art = re.sub("[^A-Za-z0-9' ]", '', str(article))
    art2 = re.sub("[( ' )(' )( ')]", ' ', str(art))
    art3 = re.sub("\s[A-Za-z]\s", ' ', str(art2))
    return art3.lower()


bow = pickle.load(open("bow.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['article']
        list_comment = [comment]
        vect = bow.transform(list_comment)

        vect = pd.DataFrame(vect.toarray())
        vect.columns = bow.get_feature_names()

        prediction_array = model.predict(vect)

    return render_template('result.html', prediction=prediction_array)


if __name__ == '__main__':
    app.run(debug=True)
    bow = pickle.load(open("bow.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
