import spacy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer

# # Chargement du modèle de langage
# nlp = spacy.load("en_core_web_sm")

# # Chargement des stopwords
# stop_words = set(stopwords.words("english"))

# # Chargement des embeddings GloVe
# glove_model = KeyedVectors.load_word2vec_format("path_to_glove_embeddings_file", binary=False)

# # Contre-ajustement des embeddings GloVe
# # Code pour le contre-ajustement des embeddings GloVe

# # Fonction pour trouver le synonyme le plus proche dans l'espace d'embedding
# def find_nearest_synonym(word):
#     if word in glove_model.vocab:
#         word_embedding = glove_model[word].reshape(1, -1)
#         _, indices = nbrs.kneighbors(word_embedding)
#         for idx in indices[0]:
#             synonym = glove_model.index2word[idx]
#             if synonym != word and synonym not in stop_words:
#                 return synonym
#     return word

# # Création du modèle de recherche du voisin le plus proche
# nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(glove_model.vectors)

# # Fonction pour remplacer les mots non-stop par des synonymes
# def word_embed_syn_swap(text):
#     doc = nlp(text)
#     new_text = []
#     for token in doc:
#         if token.text.lower() not in stop_words:
#             synonym = find_nearest_synonym(token.text.lower())
#             new_text.append(synonym if token.text.islower() else synonym.capitalize())
#         else:
#             new_text.append(token.text)
#     return " ".join(new_text)


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

# Exemple d'utilisation
# original_text = "This is a sample sentence."
# processed_text = word_embed_syn_swap(original_text)
# print("Original Text:", original_text)
# print("Processed Text:", processed_text)

# input_query = "what is durable medical equipment consist of"
# translated_query = BackTranslation(input_query)
# print("Translated Query:", translated_query)

# Il faut mettre ? à la fin de la phrase sinon ça marche pas
text = "what is durable medical equipment consist of?"
print(T5QQP(text))

