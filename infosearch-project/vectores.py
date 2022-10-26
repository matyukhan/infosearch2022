from preprocessing import preprocessing


def tfidf_vector (query, vectorizer, stopwords):
    """
    Векторизуем запрос с помощью tf-idf
    """
    prepr_query = preprocessing(query, stopwords)
    return vectorizer.transform([prepr_query])

def bm25_vector (query, vectorizer, stopwords):
    """
    Векторизуем запрос с помощью BM25
    """
    prepr_query = preprocessing(query, stopwords)
    return vectorizer.transform([prepr_query])