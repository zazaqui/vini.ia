
import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    MarianMTModel,
    MarianTokenizer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")
output_dir = "./model"
os.makedirs(output_dir, exist_ok=True)

print("[INFO] Carregando 2k exemplos do phishing-email-dataset...")
ds_en = load_dataset("zefang-liu/phishing-email-dataset", split="train[:2000]")

# 3. Uniformizador
ds_en = ds_en.rename_column("Email Text", "text")
ds_en = ds_en.rename_column("Email Type", "label")

# 4. Traduz essa bct EN→PT usando MarianMTModel (prefixo >>por<<) e truncando a 200 chars( Lucas se essa prr der erro akakka)
print("[INFO] Traduzindo 2k exemplos para pt-BR...")
mt_id        = "Helsinki-NLP/opus-mt-tc-big-en-pt"
mt_tokenizer = MarianTokenizer.from_pretrained(mt_id)
mt_model     = MarianMTModel.from_pretrained(mt_id).to("cpu")

def translate_batch(batch):
    out_texts = []
    for txt in batch["text"]:
        src = ">>por<< " + (txt[:200] if txt else "")
        inputs = mt_tokenizer(src, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            gen = mt_model.generate(**inputs)
        out_texts.append(mt_tokenizer.decode(gen[0], skip_special_tokens=True))
    return {"text_pt": out_texts}

ds_pt = ds_en.map(translate_batch, batched=True, batch_size=32)


def bin_label(ex):
    return {"labels": 0 if ex["label"]=="Phishing Email" else 1}

ds_en = ds_en.map(bin_label)
ds_pt = ds_pt.map(bin_label)

ds_pt = ds_pt.remove_columns("text")
ds_pt = ds_pt.rename_column("text_pt", "text")


print("[INFO] Combinando e dividindo datasets...")
train_en = ds_en.select(range(0, 1600))
val_en   = ds_en.select(range(1600, 2000))
train_pt = ds_pt.select(range(0, 1600))
val_pt   = ds_pt.select(range(1600, 2000))

train_ds = train_en.concatenate(train_pt).shuffle(seed=42)
val_ds   = val_en.concatenate(val_pt).shuffle(seed=42)
data = DatasetDict({"train": train_ds, "validation": val_ds})

print("[INFO] Carregando DistilBERT Multilíngue...")
model_id  = "distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_id)
model     = DistilBertForSequenceClassification.from_pretrained(
    model_id, num_labels=2
).to(device)


def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

print("[INFO] Tokenizando...")
data["train"]      = data["train"].map(tokenize_fn, batched=True, remove_columns=["text","label"])
data["validation"] = data["validation"].map(tokenize_fn, batched=True, remove_columns=["text","label"])
data["train"].set_format(type="torch", columns=["input_ids","attention_mask","labels"])
data["validation"].set_format(type="torch", columns=["input_ids","attention_mask","labels"])


def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

print("[INFO] Configurando treinamento leve...")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,                
    per_device_train_batch_size=32,     
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=0,                      
    fp16=False,                           # mixed precision para 2× mais rápido :contentReference[oaicite:7]{index=7}
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1
)

# 12. Treino
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("[INFO] Iniciando treinamento...")
trainer.train()
print("[INFO] Salvando modelo...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[INFO] Concluído.")