from nltk.corpus import stopwords
from Naturality import RemoveStopWords
import random 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from textattack.transformations import WordSwapEmbedding
from textattack.augmentation import Augmenter
from textattack.constraints.semantics import WordEmbeddingDistance



#--------------------------------------------------------------------------#

# DONE !
stop_words = set(stopwords.words('english'))


def WordEmbedSynSwap(query):

  transformation = WordSwapEmbedding()
  COSINE_DIST_CONSTRAINT = [WordEmbeddingDistance(min_cos_sim=0.9)]
  augmenter = Augmenter(transformation=transformation,transformations_per_example=1, constraints=COSINE_DIST_CONSTRAINT)


  phrase_modifiee = []

  phrases_synonyms = augmenter.augment(RemoveStopWords(query))[0].split(' ')
  index = 0
  for mot in query.split(' ') :
    if mot not in stop_words:
      phrase_modifiee.append(phrases_synonyms[index])
      index+=1
    else:
      phrase_modifiee.append(mot)

  return ' '.join(phrase_modifiee)

print(WordEmbedSynSwap("what is durable medicinal equipment consist of"))


#--------------------------------------------------------------------------#

# DONE !

from nltk.corpus import wordnet
nltk.download('punkt')

def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    print(synonyms)
    return synonyms

def WordNetSynSwap(query):
    query_cleaned = RemoveStopWords(query)
    query_splitted = query_cleaned.split()

    random_term = random.choice(query_splitted)
    synonymes =  get_synonyms(random_term)
    if len(synonymes) in [0,1]:
        return None # Non valid variation 
    else :
      print("Les synonymes : ",synonymes)
      for syn in synonymes :
          if syn != random_term :
            print("Le mot a remplacer : ",random_term ," avec ", syn)
            modified_query = query.replace(random_term, syn)
            break
      return modified_query


print(WordNetSynSwap("what is durable medicinal equipment consist of"))
