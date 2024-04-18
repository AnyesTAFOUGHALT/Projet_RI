# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')

# def RemoveStopWords(query):
#     """
#     Removes all stopwords from the query
#     """
#     query_lower = query.lower()
#     tokens = nltk.word_tokenize(query_lower)
#     query_without_stop_words = [word for word in tokens if word not in stopwords.words('english')]
#     query_without_stop_words = ' '.join(query_without_stop_words)
#     return query_without_stop_words



import ir_datasets

# dataset = ir_datasets.load("trec-robust04")
# for query in dataset.queries_iter():
#     print("----------------------------")
#     print(query) # namedtuple<query_id, title, description, narrative>
#     print("----------------------------")







# ["trec-robust04","gov2/trec-tb-2004","aquaint/trec-robust-2005","gov/trec-web-2002","clueweb12/b13/ntcir-www-2",
#  "clueweb12/b13/ntcir-www-3","clueweb12/b13/trec-misinfo-2019","cord19/trec-covid","nyt/trec-core-2017"]
# # print(RemoveStopWords("What is durable Medical equipment Consist of"))


# from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Charger le modèle T5 pré-entraîné
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Fonction pour générer des titres à partir de descriptions
# def generer_titre(description):
#     # Prétraiter la description
#     input_text = "summarize: " + description

#     # Tokenization
#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

#     # Générer le titre
#     outputs = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)

#     # Décoder et retourner le titre généré
#     titre = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return titre

# # Exemple d'utilisation
# description = "Une voiture rouge se déplaçant rapidement sur une route"
# titre_genere = generer_titre(description)
# print("Description :", description)
# print("Titre généré :", titre_genere)


# import torch
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from torch.utils.data import DataLoader, Dataset
# from ir_datasets import load

# # Charger le modèle et le tokenizer
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")

# # Charger les données de TREC Robust04
# dataset = ir_datasets.load("trec-robust04")

# # Prétraiter les données pour les transformer en paires (description -> titre)
# donnees_entrainement = [(query.description, query.title) for query in dataset.queries_iter()]

# # Définir l'ensemble de données personnalisé pour l'entraînement
# class TitreDescriptionDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# # Créer l'ensemble de données d'entraînement
# ensemble_entrainement = TitreDescriptionDataset(donnees_entrainement)

# # Définir la fonction d'entraînement
# def entrainer_modele(ensemble_entrainement, epochs=5, batch_size=8, learning_rate=1e-4):
#     optimiseur = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     dataloader = DataLoader(ensemble_entrainement, batch_size=batch_size, shuffle=True)
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for descriptions, titres in dataloader:
#             optimiseur.zero_grad()
#             input_text = ["summarize: " + desc for desc in descriptions]
#             inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#             outputs = model(**inputs, labels=tokenizer(titres, return_tensors="pt", padding=True, truncation=True).input_ids)
#             loss = outputs.loss
#             total_loss += loss.item()
#             loss.backward()
#             optimiseur.step()
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# # Entraîner le modèle
# entrainer_modele(ensemble_entrainement)

# def generer_titre(description):
#     # Prétraiter la description
#     input_text = "summarize: " + description

#     # Tokenization
#     inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

#     # Générer le titre
#     outputs = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)

#     # Décoder et retourner le titre généré
#     titre = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return titre

# # Exemple d'utilisation
# description = "Une voiture rouge se déplaçant rapidement sur une route"
# titre_genere = generer_titre(description)
# print("Description :", description)
# print("Titre généré :", titre_genere)
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned T5 model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_title(description):
    # Preprocess the input description
    input_text = "generate title: " + description + " </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate title
    output_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    
    # Decode and return the generated title
    generated_title = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_title

# Example usage
description = "Evidence that rap music has a negative effect on young people."
generated_title = generate_title(description)
print("Generated Title:", generated_title)