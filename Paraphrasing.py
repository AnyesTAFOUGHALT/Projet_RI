import gensim.downloader as api 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import random 

# # Charger le GloVe embeddings
word_vectors = api.load("glove-wiki-gigaword-300")

#--------Je l'ai juste recipier pour eviter de l'importer -------#
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

#_______________________________________________________________#


#---------------- Cette fonction Ne marche pas a 100% il dit toujours qu'il ne trouve pas de synonyme---#
def find_nearest_neighbor(word_vectors,word_vector):
    print("find_nearest_neighbor")
    # On charge égalemnt les stop words
    stop_words = set(stopwords.words('english'))
    
    nearest_neighbor = None
    min_dist = np.inf

    for word in word_vectors.index_to_key :
        if word not in stop_words : 
            similarity = cosine_similarity([word_vectors[word]],[word_vector])

            # Update nearest neighbor if closer
            if similarity > min_dist:
                nearest_neighbor = word
                min_dist = similarity
    return nearest_neighbor


def WordEmbedSynSwap(query):
    """
    Le but ici est de selctionner le manière random un mot a remplacer avec son synonyme
    """ 
    print("WordEmbedSynSwap")

    modified_query = query


    query_cleaned = RemoveStopWords(query)
    query_splitted = query_cleaned.split()
    while(True):

        random_term = random.choice(query_splitted)
        print("_______",random_term)
        if random_term in word_vectors :
            word_vector = word_vectors[random_term]
            synonyme = find_nearest_neighbor(word_vectors,word_vector)
            modified_query = query.replace(random_term, synonyme)
            break

    modified_query



print(WordEmbedSynSwap("what is durable medicinal equipment consist of"))


#--------------------------------------------------------------------------#

# J'ai une remarque pour les mots sans synonymes rediscuter 
# Le premier mot de la liste des synonymes est lui même

from nltk.corpus import wordnet

def get_first_synonym(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    print(synonyms)
    return synonyms[0]

def WordNetSynSwap(query):
    query_cleaned = RemoveStopWords(query)
    query_splitted = query_cleaned.split()

    random_term = random.choice(query_splitted)
    synonyme =  get_first_synonym(random_term)
    if synonyme == None :
        return None # Non valid variation 
    else :
        print("Le mot a remplacer : ",random_term)
        modified_query = query.replace(random_term, synonyme)

        return modified_query


print(WordNetSynSwap("what is durable medicinal equipment consist of"))
