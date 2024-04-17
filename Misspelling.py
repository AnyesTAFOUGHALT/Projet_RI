import random
import string


def RandomCharSub (query):
    """
    Replaces a random character from a random query
    term (excluding stopwords) with a randomly chosen new ASCII
    character.
    """
    query_cleaned = (query)

    query_splitted = query_cleaned.split()

    random_term = random.choice(query_splitted)
    random_char = random.choice(range(len(random_term)))

    new_char= random.choice(string.ascii_letters)

    modified_term = random_term[:random_char] + new_char + random_term[random_char + 1:]
    modified_query = query.replace(random_term, modified_term)

    return modified_query

