from Naturality import RemoveStopWords
import random

def NeighbCharSwap(query):
    """
        NeighbCharSwap Swaps two neighbouring characters from a random query term 
    """
    #query_without_stop_words = RemoveStopWords(query)
    query_without_stop_words = query
    words =  query_without_stop_words.split()
    random_term = random.choice(words)
    position = random.randint(0, len(random_term) - 2)
    term_list = list(random_term)
    term_list[position], term_list[position + 1] = term_list[position + 1], term_list[position]
    modified_term = ''.join(term_list)
    modified_query = query.replace(random_term, modified_term)
    
    return modified_query

def QWERTYCharSub(query) :
    """
    Replaces a random character of a random query
    term (excluding stopwords) with another character from the QWERTY keyboard such that only characters in close proximity are
    chosen, replicating errors that come from typing too quickly.
    """

print(NeighbCharSwap("durable medical equipment consist"))

