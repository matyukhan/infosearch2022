def create_tfidf(text, vectorizer):
    return vectorizer.fit_transform(text)