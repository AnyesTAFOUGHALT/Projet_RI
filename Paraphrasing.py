import spacy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

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

# Exemple d'utilisation
# original_text = "This is a sample sentence."
# processed_text = word_embed_syn_swap(original_text)
# print("Original Text:", original_text)
# print("Processed Text:", processed_text)

input_query = "what is durable medical equipment consist of"
translated_query = BackTranslation(input_query)
print("Translated Query:", translated_query)
