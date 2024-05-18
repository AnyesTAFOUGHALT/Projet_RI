import random
from Naturality import RemoveStopWords



def RandomOrderSwap(queries):
    """Randomly swap two words of the query."""

    queries_variations= []
    for query in queries :
      if len(query) >= 2:
        query_splitted = query.split()

        indice1, indice2 = random.sample(range(len(query_splitted)), 2)

        print(query_splitted[indice1] , query_splitted[indice2])

        query_splitted[indice1] , query_splitted[indice2] = query_splitted[indice2] , query_splitted[indice1]
        
        modified_query =  ' '.join(query_splitted)
        queries_variations.append(modified_query)
      else :
        queries_variations.append(query)
      
    return queries_variations
