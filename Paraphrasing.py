import gensim.downloader as api 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import random 
import spacy
from gensim.models import KeyedVectors
from sklearn.neighbors import NearestNeighbors
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer

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

#---------------------------------------------------------------------------#

def BackTranslation(input_query, pivot_language='de', original_language='en'):
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    tokenizer.src_lang = original_language
    encoded_en = tokenizer(input_query, return_tensors="pt")
    generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(pivot_language))
    translated_query = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    translated_query
    
    tokenizer.src_lang = pivot_language
    encoded_de = tokenizer(translated_query, return_tensors="pt")
    generated_tokens = model.generate(**encoded_de, forced_bos_token_id=tokenizer.get_lang_id(original_language))
    back_translated_query = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return back_translated_query

# input_query = "what is durable medical equipment consist of"
# translated_query = BackTranslation(input_query)
# print("Translated Query:", translated_query)

#---------------------------------------------------------------------------#

def T5QQP(input_query, max_length=128):
    # Chargement du tokenizer et du modèle fine-tuned pour la génération de paraphrases de questions
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")

    # Encodage du texte en utilisant le tokenizer
    input_ids = tokenizer.encode(input_query, return_tensors="pt", add_special_tokens=True)

    # Génération des paraphrases
    generated_ids = model.generate(input_ids=input_ids, 
                                    num_return_sequences=2, 
                                    num_beams=5, 
                                    max_length=max_length, 
                                    no_repeat_ngram_size=2, 
                                    repetition_penalty=3.5, 
                                    length_penalty=1.0, 
                                    early_stopping=True)

    # Décodage des paraphrases générées
    paraphrase = tokenizer.decode(generated_ids[1], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return paraphrase

# Il faut mettre ? à la fin de la phrase sinon ça marche pas
text = "what is durable medical equipment consist of?"
print(T5QQP(text))