def measure_similarity(matrix, vector):
    """
    Считаем косинусную близость запроса и документов корпуса
    """
    return matrix.dot(vector.T)
