import random
from Naturality import RemoveStopWords



def RandomOrderSwap(queries):
    """Randomly swap two words of the query."""

    queries_variations= []
    for query in queries :
        query_cleaned = RemoveStopWords(query)

        if len(query) >= 2:
            query_splitted = query_cleaned.split()

            indice1, indice2 = random.sample(range(len(query_splitted)), 2)
            query_splitted[indice1],query_splitted[indice2] = query_splitted[indice2],query_splitted[indice1]

            query_cleaned =  ' '.join(query_splitted)
        
        queries_variations.append(query_cleaned)
    return queries_variations
