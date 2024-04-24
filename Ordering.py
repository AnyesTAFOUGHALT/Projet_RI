import random
from Naturality import RemoveStopWords



def RandomOrderSwap(query):
    """Randomly swap two words of the query."""
    query_cleaned = (query)# remove stopwords?

    if len(query) >= 2:
        query_splitted = query_cleaned.split()

        indice1, indice2 = random.sample(range(len(query_splitted)), 2)
        query_splitted[indice1],query_splitted[indice2] = query_splitted[indice2],query_splitted[indice1]

        query_cleaned =  ' '.join(query_splitted)
    
    return query_cleaned


# for i in range(10):
#     print(RandomOrderSwap("what is durable mdeical equipment consist of"))