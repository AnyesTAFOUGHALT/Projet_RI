
import random 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer

from nltk.corpus import stopwords
from Naturality import RemoveStopWords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.corpus import wordnet
nltk.download('punkt')

from textattack.transformations import WordSwapEmbedding
from textattack.augmentation import Augmenter
from textattack.constraints.semantics import WordEmbeddingDistance

stop_words = set(stopwords.words('english'))

#--------------------------------------------------------------------------#

# DONE !



def WordEmbedSynSwap(queries):

    transformation = WordSwapEmbedding()
    COSINE_DIST_CONSTRAINT = [WordEmbeddingDistance(min_cos_sim=0.9)]
    augmenter = Augmenter(transformation=transformation,transformations_per_example=1, constraints=COSINE_DIST_CONSTRAINT)

    queries_variations= []
    for query in queries :
        phrase_modifiee = []

        phrases_synonyms = augmenter.augment(RemoveStopWords(query))[0].split(' ')
        index = 0
        for mot in query.split(' ') :
            if mot not in stop_words:
                phrase_modifiee.append(phrases_synonyms[index])
                index+=1
            else:
                phrase_modifiee.append(mot)

        queries_variations.append(' '.join(phrase_modifiee))
    return queries_variations


# print(WordEmbedSynSwap("what is durable medicinal equipment consist of"))


#--------------------------------------------------------------------------#

# DONE !


def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())

    print(synonyms)
    return synonyms

def WordNetSynSwap(queries):
    queries_variations= []
    for query in queries :
        query_cleaned = RemoveStopWords(query)
        query_splitted = query_cleaned.split()

        random_term = random.choice(query_splitted)
        synonymes =  get_synonyms(random_term)
        if len(synonymes) in [0,1]:
            queries_variations.append(None) # Non valid variation 
        else :
            print("Les synonymes : ",synonymes)
            for syn in synonymes :
                if syn != random_term :
                    print("Le mot a remplacer : ",random_term ," avec ", syn)
                    modified_query = query.replace(random_term, syn)
                    break
            queries_variations.append(modified_query)
    return queries_variations


print(WordNetSynSwap("what is durable medicinal equipment consist of"))

#---------------------------------------------------------------------------#

def BackTranslation(queries, pivot_language='de', original_language='en'):
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    queries_variations= []
    for input_query in queries :
        tokenizer.src_lang = original_language
        encoded_en = tokenizer(input_query, return_tensors="pt")
        generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(pivot_language))
        translated_query = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated_query
        
        tokenizer.src_lang = pivot_language
        encoded_de = tokenizer(translated_query, return_tensors="pt")
        generated_tokens = model.generate(**encoded_de, forced_bos_token_id=tokenizer.get_lang_id(original_language))
        back_translated_query = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        queries_variations.append(back_translated_query)

    return queries_variations

# input_query = "what is durable medical equipment consist of"
# translated_query = BackTranslation(input_query)
# print("Translated Query:", translated_query)

#---------------------------------------------------------------------------#

def T5QQP(queries, max_length=128):
    # Chargement du tokenizer et du modèle fine-tuned pour la génération de paraphrases de questions
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")


    queries_variations= []
    for input_query in queries :
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

        queries_variations.append(paraphrase)
    return queries_variations

# Il faut mettre ? à la fin de la phrase sinon ça marche pas
text = "what is durable medical equipment consist of?"
print(T5QQP(text))