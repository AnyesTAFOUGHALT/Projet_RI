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





"""
Ici pour resumer j'ai fintuner sur tous les dastes qu'ils ont mentionné la seule remarque est que j'ai utilise 
que 3 epoch deja car ca prend enormement de temps
et regrder tout en bas le test que j'ai fait => ce que ça a retouner (Est ce qu'on est sensé avoir 
éxacetement de même résultat)


"""
import ir_datasets
count = 0
titles = []
descriptions = []
for dtst in ["disks45/nocr/trec-robust-2004","gov2/trec-tb-2004","aquaint/trec-robust-2005", "gov/trec-web-2002", "clueweb12/b13/ntcir-www-2", "clueweb12/b13/ntcir-www-3",
              "clueweb12/b13/trec-misinfo-2019", "cord19/trec-covid" ,"nyt/trec-core-2017"]:
    dataset = ir_datasets.load(dtst)
    for query in dataset.queries_iter():
        titles.append(query.title)
        descriptions.append(query.description)

print("En tout j'aurai : ",len(titles), " titres")
print("En tout j'aurai : ",len(descriptions), " descriptions")
print(" Titre : ",titles[100], " descriptions : ", descriptions[100])


import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, descriptions, titles, tokenizer, max_length=128):
        self.descriptions = descriptions
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        title = self.titles[idx]

        input_text = "summarize: " + description
        target_text = title

        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")



        # Reshape labels to (batch_size, seq_length)
        input_ids = input_ids.reshape(-1, 128)
        target_ids = target_ids.reshape(-1, 128)

        return {"input_ids": input_ids, "labels": target_ids}

# Define your fine-tuning function
def fine_tune_t5(train_dataset, val_dataset, tokenizer, model_name="t5-small", num_epochs=3, batch_size=8, learning_rate=3e-4):
    # Load pre-trained T5 model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)
            input_ids = input_ids.view(batch_size, -1)
            labels = labels.view(batch_size, -1)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                batch_size = input_ids.size(0)
                input_ids = input_ids.view(batch_size, -1)
                labels = labels.view(batch_size, -1)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                total_val_loss += loss.item()

        print(f"Epoch {epoch + 1}: Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {total_val_loss / len(val_loader):.4f}")

        # Adjust learning rate
        scheduler.step()

    return model

# Example usage
train_descriptions = descriptions[:600]  # List of training descriptions
train_titles = titles[:600]        # List of corresponding training titles
val_descriptions = descriptions[600:]    # List of validation descriptions
val_titles = titles[600:]          # List of corresponding validation titles

tokenizer = T5Tokenizer.from_pretrained("t5-small")

train_dataset = CustomDataset(train_descriptions, train_titles, tokenizer)
val_dataset = CustomDataset(val_descriptions, val_titles, tokenizer)

# Fine-tune the T5 model
fine_tuned_model = fine_tune_t5(train_dataset, val_dataset, tokenizer)


# Example description
description = "What is durable medical equipement consist of ."

# Tokenize the description

inputs = tokenizer.encode("summarize: " + description, return_tensors="pt")

# Generate title
outputs = fine_tuned_model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)

# Decode and print the generated title
title = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Title:", title)


#===> Generated Title: medical equipement durable
