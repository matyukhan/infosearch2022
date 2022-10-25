import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from create_tfidf import create_tfidf
from create_bm25 import create_bm25
from preprocessing import preprocessing

def save_matrix():
    with open("data.jsonl", "r") as data:
        raw_text = list(data[:50000])
        questions = []
        for t in raw_text:
            text = json.loads(t)
            question = text["questions"]
            questions.append(preprocessing(question, sw))

    cv = CountVectorizer()
    tf = TfidfVectorizer()
    tf_bm25 = TfidfVectorizer()

    matrix_tfidf = create_tfidf(questions, tf)
    with open("matrix_tfidf.pickle", "wb") as m:
        pickle.dump(matrix_tfidf, m)
    with open("vectorizer_tfidf.pickle", "wb") as v:
        pickle.dump(tf, v)

    matrix_bm25 = create_bm25(questions, cv, tf_bm25)
    with open("matrix_bm25.pickle", "wb") as m:
        pickle.dump(matrix_bm25, m)
    with open("vectorizer_bm25_cv.pickle", "wb") as v_cv:
        pickle.dump(cv, v_cv)
    with open("vectorizer_bm25_tf.pickle", "wb") as v_tf:
        pickle.dump(tf_bm25, v_tf)