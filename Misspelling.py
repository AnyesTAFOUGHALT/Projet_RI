import random
import string
from Naturality import RemoveStopWords
import re

def suppression_ponctuation(text):
    punc = string.punctuation  
    punc += '\n\r\t'
    text = text.translate(str.maketrans(punc, ' ' * len(punc)))  
    text = re.sub('( )+', ' ', text)

    return text

keyboard_layout = {
    'q': ['w', 'a', 's'],
    'w': ['q', 'a', 's', 'd', 'e'],
    'e': ['w', 's', 'd', 'f', 'r'],
    'r': ['e', 'd', 'f', 'g', 't'],
    't': ['r', 'f', 'g', 'h', 'y'],
    'y': ['t', 'g', 'h', 'j', 'u'],
    'u': ['y', 'h', 'j', 'k', 'i'],
    'i': ['u', 'j', 'k', 'l', 'o'],
    'o': ['i', 'k', 'l', 'p'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'],
    's': ['q', 'w', 'e', 'a', 'd', 'z', 'x'],
    'd': ['w', 'e', 'r', 's', 'f', 'x', 'c'],
    'f': ['e', 'r', 't', 'd', 'g', 'c', 'v'],
    'g': ['r', 't', 'y', 'f', 'h', 'v', 'b'],
    'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n'],
    'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm'],
    'k': ['u', 'i', 'o', 'j', 'l', 'm'],
    'l': ['i', 'o', 'p', 'k'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k']
}

def NeighbCharSwap(queries):
    """
        NeighbCharSwap Swaps two neighbouring characters from a random query term 
    """
    queries_variations= []
    for query in queries :
        query_without_stop_words = RemoveStopWords([query])[0]
        query_without_stop_words = suppression_ponctuation(query_without_stop_words)
        words =  query_without_stop_words.split()
        if len(words) != 0:
          while True:
            random_term = random.choice(words)
            if len(random_term) >= 2:
              break
          print(random_term)
          position = random.randint(0, len(random_term) - 2)
          term_list = list(random_term)
          term_list[position], term_list[position + 1] = term_list[position + 1], term_list[position]
          modified_term = ''.join(term_list)
          modified_query = query.replace(random_term, modified_term)
          
          queries_variations.append(modified_query)
        else :
          queries_variations.append(query)
    return queries_variations



def RandomCharSub (queries):
    """
    Replaces a random character from a random query
    term (excluding stopwords) with a randomly chosen new ASCII
    character.
    """

    queries_variations= []
    for query in queries :
        query_cleaned = RemoveStopWords([query])[0]
        query_cleaned = suppression_ponctuation(query_cleaned)
        query_splitted = query_cleaned.split()
        if len(query_splitted) != 0 :
          random_term = random.choice(query_splitted)
          random_char = random.choice(range(len(random_term)))

          new_char= random.choice(string.ascii_letters)

          modified_term = random_term[:random_char] + new_char + random_term[random_char + 1:]
          modified_query = query.replace(random_term, modified_term)

          queries_variations.append(modified_query)
        else :
          queries_variations.append(query)
    return queries_variations



def QWERTYCharSub(queries) :
    """
    Replaces a random character of a random query
    term (excluding stopwords) with another character from the QWERTY keyboard such that only characters in close proximity are
    chosen, replicating errors that come from typing too quickly.
    """

    queries_variations= []
    for query in queries :
        query_without_stop_words = RemoveStopWords([query])[0]
        query_without_stop_words = suppression_ponctuation(query_without_stop_words)
        words =  query_without_stop_words.split()
        if len(words) != 0 :
          while True :
            random_term = random.choice(words)
            position = random.randint(0, len(random_term) - 1)
            term_list = list(random_term)
            if term_list[position] in keyboard_layout.keys():
              new_char = random.choice(keyboard_layout[term_list[position]])
              term_list[position] = new_char
              modified_term = ''.join(term_list)
              modified_query = query.replace(random_term, modified_term)
              
              queries_variations.append(modified_query)
              break
        else :
          queries_variations.append(query)

    return queries_variations   