import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def RemoveStopWords(query):
    """
    Removes all stopwords from the query
    """
    query_lower = query.lower()
    tokens = nltk.word_tokenize(query_lower)
    query_without_stop_words = [word for word in tokens if word not in stopwords.words('english')]
    query_without_stop_words = ' '.join(query_without_stop_words)
    return query_without_stop_words

print(RemoveStopWords("What is durable Medical equipment Consist of"))