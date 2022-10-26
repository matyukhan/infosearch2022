import pickle
import json
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from create_metrics import create_tfidf
from create_metrics import create_bm25
from preprocessing import preprocessing
from nltk.corpus import stopwords
from tqdm import tqdm

sw = stopwords.words('russian')


def save_matrix():
    """
    Векторизуем текст корпуса и сохраняем в матрицу
    """
    with open("data.jsonl", "r") as data:
        raw_text = list(data)[:50000]
        questions = []
        questions_wth_prepr = []
        for t in tqdm(raw_text):
            text = json.loads(t)
            question = text["question"]
            questions_wth_prepr.append(question)
            questions.append(preprocessing(question, sw))

    with open("data.txt", "w") as d:
        d.write('\n'.join(questions))

    with open("data_wth_prepr.txt", "w") as d:
        d.write('\n'.join(questions_wth_prepr))

    cv = CountVectorizer()
    tf = TfidfVectorizer()
    tf_bm25 = TfidfVectorizer()

    matrix_tfidf = create_tfidf(questions, tf)
    with open("matrix_tfidf.pickle", "wb") as m:
        pickle.dump(matrix_tfidf, m)
    with open("vectorizer_tfidf.pickle", "wb") as v:
        pickle.dump(tf, v)
    print('TF-IDF saved')

    matrix_bm25 = create_bm25(questions, cv, tf_bm25)
    with open("matrix_bm25.pickle", "wb") as m:
        pickle.dump(matrix_bm25, m)
    with open("vectorizer_bm25_cv.pickle", "wb") as v_cv:
        pickle.dump(cv, v_cv)
    with open("vectorizer_bm25_tf.pickle", "wb") as v_tf:
        pickle.dump(tf_bm25, v_tf)
    print('BM25 saved')

save_matrix()