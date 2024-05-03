import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from transformers import pipeline


def RemoveStopWords(query):
    """
    Removes all stopwords from the query
    """
    query_lower = query.lower()
    tokens = nltk.word_tokenize(query_lower)
    query_without_stop_words = [word for word in tokens if word not in stopwords.words('english')]
    query_without_stop_words = ' '.join(query_without_stop_words)
    return query_without_stop_words



def T5DescToTitle(queries):
  summarizer = pipeline("summarization", model="fine_tunned-models/t5-base_from_description_to_title/", tokenizer="t5-base")

  query_variations = []
  titles = summarizer(queries, min_length=3, max_length=8)
  for title in titles:
      query_variations.append(title)
      
  return query_variations

T5DescToTitle(["How many civilian non-combatants have been killed in \nthe various civil wars in Africa?"])
# ==> Africa civil wars 
"""
[['what is durable medical equipment consist of.',
  'durable medical equipment',
  'summarization_with_t5-base_from_description_to_title',
  'naturality']]
  
"""