from pymorphy2 import MorphAnalyzer
import string

def preprocessing(text, sw):
    """
    Токенизируем текст, приводим к нижнему регистру и оставляем только последовательности из букв,
    лемматизируем и удаляем стоп-слова
    """
    morph = MorphAnalyzer()
    punct = string.punctuation

    words = [w.lower().strip(punct) for w in text.split() if w.isalpha()]
    filtered = [w for w in words if w not in sw]
    tokens = [morph.parse(w)[0].normal_form for w in filtered]
    return ' '.join(tokens)