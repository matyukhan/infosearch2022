# Финальный проект: поисковик с помощью TF-IDF и BM25
Идея этого проекта в том, что с помощью поисковика можно лишний раз убедиться, что все вопросы о любви лишь ипостаси тысячеликого героя и новых вопросов придумать невозможно. В этом можно убедиться, задав любой вопрос в поисковой строке и получив аналогичный вопрос в выдаче.
*Язык: Python
*Исходный датасет: корпус ответов майл.ру о любви

## Как запускать проект:
0. Убедитесь, что у вас установлены все библиотеки из файла `requirements.txt`
1. Запустите `save_matrix.py`. После (длительной) обработки в проекте появятся .pickle файлы.
2. После появления файлов запустите `main.py`
3. Запустите приложение streamlit в Терминале из папки `infosearch-project` следующей командой:
```
python3 -m streamlit run main.py
```
