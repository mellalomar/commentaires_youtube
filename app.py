from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
app = Flask(__name__)


@app.route('/')
def home():
    nltk.download('punkt')
    nltk.download('wordnet')

    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/Eminem.csv")
    df_data = df[['CONTENT', 'CLASS']]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract the features with countVectorizer
    corpus = df_x
    corpus_clean = []
    for txt in corpus:
        word_tokens = [word.lower() for word in word_tokenize(txt)]
        clean_words = [word for word in word_tokens if (
            not word in set(stopwords.words('english')) and word.isalpha())]
        lemmmatizer = WordNetLemmatizer()
        clean_words = [lemmmatizer.lemmatize(
            word.lower()) for word in clean_words]
        corpus_clean.append(' '.join(clean_words))
    cv = CountVectorizer()
    X = cv.fit_transform(corpus_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, df_y, test_size=0.33, random_state=42)
    # Navie Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
