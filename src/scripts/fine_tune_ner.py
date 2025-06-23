import re
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from ner_wrapper import NERWrapper
from lime.lime_text import LimeTextExplainer

# --- STEP 1: Define Labels ---
label_list = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# --- STEP 2: Load CoNLL-style text files ---
def read_conll(filepath):
    """
    Reads a CoNLL-formatted file and parses it into a list of samples for NER tasks.
    Each sample is represented as a dictionary with two keys:
        - "tokens": a list of token strings from a sentence.
        - "ner_tags": a list of integer NER tag IDs corresponding to each token.
    The function expects each line in the file to contain at least four whitespace-separated columns,
    where the first column is the token and the fourth column is the NER tag (as a string).
    Sentences are separated by blank lines.
    Args:
        filepath (str): Path to the CoNLL-formatted file.
    Returns:
        List[Dict[str, List]]: A list of dictionaries, each containing "tokens" and "ner_tags" for a sentence.
    """
    with open(filepath, encoding="utf-8") as f:
        tokens, tags, samples = [], [], []
        for line in f:
            
            line = line.strip()
            if line == "":
                if tokens:
                    samples.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                # line = normalize_label(line)
                splits = line.split()
                # print(splits)
                tokens.append(splits[0])
                label = normalize_label(splits[3])
                tags.append(label_to_id.get(label, 0))

        if tokens:
            samples.append({"tokens": tokens, "ner_tags": tags})
    return samples

def normalize_label(text):
    subs = {
        'B-I-Product': 'B-Product',
        'B-B-Product': 'B-Product',
        'I-B-Product': 'I-Product',
        'I-I-Product': 'I-Product',
        'B-B-PRICE': 'B-PRICE',
        'B-I-PRICE': 'B-PRICE',
        'B-B-LOC': 'B-LOC',
        'B-I-LOC': 'B-LOC',
        'I-I-LOC': 'I-LOC'
    }

    # for pattern, replacement in subs.items():
    #     text = re.sub(pattern, replacement, text)
    return subs.get(text, text)

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
    """    Tokenizes the input tokens and aligns the NER labels with the tokenized output.
    Args:
        example (Dict): A dictionary containing "tokens" (list of strings) and "ner_tags" (list of integers).
    Returns:
        Dict: A dictionary with tokenized inputs and aligned labels.
    """
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
    """
    Compute evaluation metrics for a sequence labeling task.
    Args:
        pred (Tuple): A tuple containing model predictions and true labels.
            - predictions: np.ndarray of shape (batch_size, seq_len, num_labels)
            - labels: np.ndarray of shape (batch_size, seq_len)
    Returns:
        Dict: A dictionary containing precision, recall, F1 score, and accuracy.
    """



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
trainer.save_model("./models/xlm-roberta")

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

# Initialize
ner_model = NERWrapper(model, tokenizer)
explainer = LimeTextExplainer(class_names=label_list)

text = "ዋጋ 1000 ብር በአዲስ አበባ ይሸጣል።"
# Explain
# exp = explainer.explain_instance(text, ner_model.predict_proba, num_features=10)
# exp.show_in_notebook()

