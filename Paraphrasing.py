import spacy
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors

# Chargement du modèle de langage
nlp = spacy.load("en_core_web_sm")

# Chargement des stopwords
stop_words = set(stopwords.words("english"))

# Chargement des embeddings GloVe
glove_model = KeyedVectors.load_word2vec_format("path_to_glove_embeddings_file", binary=False)

# Contre-ajustement des embeddings GloVe
# Code pour le contre-ajustement des embeddings GloVe

# Fonction pour trouver le synonyme le plus proche dans l'espace d'embedding
def find_nearest_synonym(word):
    if word in glove_model.vocab:
        word_embedding = glove_model[word].reshape(1, -1)
        _, indices = nbrs.kneighbors(word_embedding)
        for idx in indices[0]:
            synonym = glove_model.index2word[idx]
            if synonym != word and synonym not in stop_words:
                return synonym
    return word

# Création du modèle de recherche du voisin le plus proche
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(glove_model.vectors)

# Fonction pour remplacer les mots non-stop par des synonymes
def word_embed_syn_swap(text):
    doc = nlp(text)
    new_text = []
    for token in doc:
        if token.text.lower() not in stop_words:
            synonym = find_nearest_synonym(token.text.lower())
            new_text.append(synonym if token.text.islower() else synonym.capitalize())
        else:
            new_text.append(token.text)
    return " ".join(new_text)

# Exemple d'utilisation
original_text = "This is a sample sentence."
processed_text = word_embed_syn_swap(original_text)
print("Original Text:", original_text)
print("Processed Text:", processed_text)
