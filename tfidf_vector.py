def tfidf_vector (query, vectorizer, stopwords):
    return vectorizer.transform(preprocessing(query, stopwords).split(" "))