import pickle
import numpy as np
from nltk.corpus import stopwords
from measure_similarity import measure_similarity
from vectores import tfidf_vector
from vectores import bm25_vector
import streamlit as st
import base64
import time

sw = stopwords.words("russian")

#передаем матрицы и векторы в переменные
with open("matrix_tfidf.pickle", "rb") as f:
    tfidf_matrix = pickle.load(f)
with open("vectorizer_tfidf.pickle", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("matrix_bm25.pickle", "rb") as f:
    bm25_matrix = pickle.load(f)
with open("vectorizer_bm25_tf.pickle", "rb") as f:
    bm25_tf_vectorizer = pickle.load(f)
with open("vectorizer_bm25_cv.pickle", "rb") as f:
    bm25_cv_vectorizer = pickle.load(f)

with open("data_wth_prepr.txt", "r", encoding="utf-8") as d:
    corpus = d.readlines()

def add_bg_from_local(image_file):
    """Установить фон для страницы"""
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.jpg')


# подключаем streamlit
st.title("Здесь можно задать любой вопрос о любви")
output = st.number_input(
    "Количество ответов",
    min_value=1,
    max_value=50000,
    help="""чтобы понять, что все вопросы о любви неоригинальны. Выберите, сколько похожих вопросов вы хотите просмотреть (от 1 до 50000)""",
)
query = st.text_input('Введите ваш вопрос в строку ниже')
left_col, right_col, third_column = st.columns(3)
model_type = left_col.selectbox('', ['TF-IDF', 'BM25'])
if st.button('Найти единомышленников'):
    query_time = time.time()
    if model_type == 'TF-IDF':
        similarity = measure_similarity(tfidf_matrix, tfidf_vector(query, tfidf_vectorizer, sw))
    elif model_type == 'BM25':
        similarity = measure_similarity(bm25_matrix, bm25_vector(query, bm25_cv_vectorizer, sw))

    # сортировка спарс-матрицы
    rows = similarity.nonzero()[0]
    row_value = zip(rows, similarity.data)
    sorted_row_value = sorted(row_value, key=lambda v: v[1], reverse=True)
    sorted_similarity_index = [i[0] for i in sorted_row_value][:output]
    result = np.array(corpus)[sorted_similarity_index]
    search_time = time.time() - query_time
    st.markdown(f'На то, чтобы найти похожие вопросы, нам понадобилось {round(search_time, 3)} секунд. А сколько понадобится вам, чтобы найти на них ответ?')
    for r in result:
        st.markdown(f'* {r}')



