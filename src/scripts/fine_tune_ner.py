# !pip install transformers datasets seqeval --quiet

import os
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shap

# --- STEP 1: Define Labels ---
label_list = ["O", "B-I-Product", "B-B-Product", "I-B-Product", "I-I-Product",
               "B-B-PRICE", "B-I-PRICE", "B-B-LOC", "B-I-LOC", "I-I-LOC"]
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# --- STEP 2: Load CoNLL-style text files ---
def read_conll(filepath):
    with open(filepath, encoding="utf-8") as f:
        tokens, tags, samples = [], [], []
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    samples.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                splits = line.split()
                # print(splits)
                tokens.append(splits[0])
                tags.append(label_to_id.get(splits[3], 0))
        if tokens:
            samples.append({"tokens": tokens, "ner_tags": tags})
    return samples

train_data = read_conll("./data/sample_label.txt") 
valid_data = read_conll("./data/sample_label.txt")

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(valid_data)
})

# data_files = {"train": "../../data/sample_label.txt", "validation": "../../data/sample_label.txt"}
# dataset = load_dataset("text", data_files=data_files)

# --- STEP 3: Load tokenizer ---
model_checkpoint = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# --- STEP 4: Tokenize and align labels ---
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            aligned_labels.append(example["ner_tags"][word_id])
        else:
            # If subword, mark as I-*
            label = example["ner_tags"][word_id]
            if label == 0:
                aligned_labels.append(0)
            else:
                aligned_labels.append(label)
        prev_word_id = word_id
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

# --- STEP 5: Load model ---
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# --- STEP 6: Define training arguments ---
training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    save_strategy="no",
    eval_strategy="no",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# --- STEP 7: Define metrics ---
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id_to_label[p] for (p, l) in zip(preds, label) if l != -100]
                  for preds, label in zip(predictions, labels)]

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds)
    }

# --- STEP 8: Fine-tune the model ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()
# trainer.save_model("./models")

print("✅ Model fine-tuned and saved!")

# After predictions
output = trainer.predict(tokenized_dataset["validation"])
predictions = output.predictions
labels = output.label_ids
preds = np.argmax(predictions, axis=2)

true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
true_preds = [[id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
              for pred, label in zip(preds, labels)]

# Generate classification report
report = classification_report(true_labels, true_preds)

print(report)
# # Extract metrics
# entity_types = [label for label in report.keys() if label not in ["micro avg", "macro avg", "weighted avg", "accuracy"]]
# f1_scores = [report[ent]["f1-score"] for ent in entity_types]
# precisions = [report[ent]["precision"] for ent in entity_types]
# recalls = [report[ent]["recall"] for ent in entity_types]

# # Plot
# x = range(len(entity_types))
# plt.figure(figsize=(10, 6))
# plt.bar(x, precisions, width=0.25, label="Precision", align="center")
# plt.bar([i + 0.25 for i in x], recalls, width=0.25, label="Recall", align="center")
# plt.bar([i + 0.50 for i in x], f1_scores, width=0.25, label="F1-Score", align="center")
# plt.xticks([i + 0.25 for i in x], entity_types, rotation=45)
# plt.ylabel("Score")
# plt.title("NER Evaluation Metrics per Entity Type")
# plt.legend()
# plt.tight_layout()
# plt.show()

explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(tokenized_dataset["validation"][:10]["tokens"])

shap.plots.text(shap_values[0])