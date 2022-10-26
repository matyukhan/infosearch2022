from scipy import sparse


def create_bm25(text, count_vectorizer, tfidf_vectorizer):
    k = 2
    b = 0.75

    cv = count_vectorizer.fit_transform(text)
    tf = tfidf_vectorizer.fit_transform(text)
    idf = tfidf_vectorizer.idf_

    len_d = tf.sum(axis=1)
    avgl = len_d.mean()

    values_a = []
    rows_a = []
    cols_a = []

    for i, j in zip(*cv.nonzero()):
        values_a.append(cv[i, j] * idf[j] * (k + 1))
        rows_a.append(i)
        cols_a.append(j)
    A = sparse.csr_matrix((values_a, (rows_a, cols_a)))

    B_1 = (k * (1 - b + b * len_d / avgl))

    values = []
    rows = []
    cols = []

    for i, j in zip(*cv.nonzero()):
        values.append(float(A[i, j] / (B_1[i] + tf[i, j]).tolist()[0][0]))
        rows.append(i)
        cols.append(j)
    bm25 = sparse.csr_matrix((values, (rows, cols)))
    return bm25


def create_tfidf(text, vectorizer):
    return vectorizer.fit_transform(text)