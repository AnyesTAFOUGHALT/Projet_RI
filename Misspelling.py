import random
import string
from Naturality import RemoveStopWords

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

def NeighbCharSwap(query):
    """
        NeighbCharSwap Swaps two neighbouring characters from a random query term 
    """
    query_without_stop_words = RemoveStopWords(query)
    words =  query_without_stop_words.split()
    random_term = random.choice(words)
    position = random.randint(0, len(random_term) - 2)
    term_list = list(random_term)
    term_list[position], term_list[position + 1] = term_list[position + 1], term_list[position]
    modified_term = ''.join(term_list)
    modified_query = query.replace(random_term, modified_term)
    
    return modified_query



def RandomCharSub (query):
    """
    Replaces a random character from a random query
    term (excluding stopwords) with a randomly chosen new ASCII
    character.
    """
    query_cleaned = RemoveStopWords(query)

    query_splitted = query_cleaned.split()

    random_term = random.choice(query_splitted)
    random_char = random.choice(range(len(random_term)))

    new_char= random.choice(string.ascii_letters)

    modified_term = random_term[:random_char] + new_char + random_term[random_char + 1:]
    modified_query = query_cleaned.replace(random_term, modified_term)

    return modified_query


def QWERTYCharSub(query) :
    """
    Replaces a random character of a random query
    term (excluding stopwords) with another character from the QWERTY keyboard such that only characters in close proximity are
    chosen, replicating errors that come from typing too quickly.
    """
    query_without_stop_words = RemoveStopWords(query)
    words =  query_without_stop_words.split()
    random_term = random.choice(words)
    position = random.randint(0, len(random_term) - 1)
    term_list = list(random_term)
    new_char = random.choice(keyboard_layout[term_list[position]])
    term_list[position] = new_char
    modified_term = ''.join(term_list)
    modified_query = query.replace(random_term, modified_term)
    
    return modified_query

print(QWERTYCharSub("durable medical equipment consist"))
print(RandomCharSub ("what is durable medical equipment consist of"))