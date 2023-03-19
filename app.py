import os
import nltk
from flask import Flask, request, render_template, url_for, redirect
import docx
import pandas as pd
import numpy as np
import re 
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)



def clean_text(text):
    txt = re.sub("[^a-zA-Z]",  # Search for all non-letters
                          " ",          # Replace all non-letters with spaces
                          str(text))
    txt_tokens=[word for word in word_tokenize(txt)if word not in string.punctuation]

     #Import the english stop words list from NLTK
    stm= LancasterStemmer()
    txt=' '.join([stm.stem(word) for word in txt_tokens if word not in stopwords.words('english') ])
  
    return txt


def cosim(x, y):

    corpus = [clean_text(x),clean_text(y)]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(corpus)
    cosm= round(cosine_similarity(X_train_counts)[1][0] * 100,2)
    return cosm





@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process_text', methods=['POST'])
def process_text():
    if request.method == 'POST':
            
        text1 = request.form['text1']
        text2= request.form['text2']
        ratio=cosim(text1, text2)
        rslt= 'Match Rate: {} %'.format(str(ratio))
        return redirect(url_for('result', rslt= rslt))
    else:
        return render_template('index.html')


@app.route('/<rslt>')
def result(rslt):
    
    return render_template('result.html', scr= rslt)


if __name__ == "__main__":
    app.run(debug= True)