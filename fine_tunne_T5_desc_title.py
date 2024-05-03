import numpy as np

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer
from datasets import Dataset
from IPython import embed
import pandas as pd
import numpy as np
import nltk
import datasets
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




# Example usage
train_descriptions = descriptions[:600]  # List of training descriptions
train_titles = titles[:600]        # List of corresponding training titles
val_descriptions = descriptions[600:]    # List of validation descriptions
val_titles = titles[600:]          # List of corresponding validation titles

# Créer un dictionnaire contenant vos données
data_dict = {"description": descriptions, "title": titles}

# Créer un objet Dataset à partir du dictionnaire
custom_dataset = Dataset.from_dict(data_dict)

max_input_length = 128
max_target_length = 64


def preprocess_function(dataset):
    prefix = "summarize: "
    inputs_desc = [prefix + doc for doc in dataset["description"]]
    model_inputs_desc = tokenizer(inputs_desc, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer([t for t in dataset["title"]], max_length=max_target_length, truncation=True)

    model_inputs_desc["labels"] = labels["input_ids"]
    return model_inputs_desc


model_checkpoint = "t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_datasets = custom_dataset.map(preprocess_function,batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def postprocess_text(preds, labels):
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    return preds, labels

metric = datasets.load_metric('sacrebleu')

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels,use_stemmer=True)
    # result = {"bleu": result["score"]}
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

batchsize= 30
training_args = Seq2SeqTrainingArguments(
  "desc-to-title-summarization-for-trec-desc",
  learning_rate=2e-4,
  per_device_train_batch_size=batchsize,
  per_device_eval_batch_size=batchsize,
  weight_decay=0.01,
  save_total_limit=3,
  num_train_epochs=6,
  predict_with_generate=True
)

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets ,
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics,
)
trainer.train()

model.save_pretrained("./fine_tunned-models/T5_from_Desc_to_Title")


# from google.colab import files

# # Chemin où vous voulez sauvegarder le modèle
# model_path = "{}_from_{}_to_{}".format(model_checkpoint, "description", "title")

# # Sauvegarder le modèle dans le répertoire actuel
# model.save_pretrained(model_path)

# # Télécharger le fichier
# files.download(model_path)