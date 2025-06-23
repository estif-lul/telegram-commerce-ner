from transformers import AutoTokenizer, AutoModelForTokenClassification

label_list = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

def predict_ner(text, model, tokenizer, id_to_label):
    # Tokenize input (split into words, then encode)
    tokens = text.split()  # Or use a better tokenizer for Amharic if available
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    # Map predictions to labels, skipping special tokens
    word_ids = inputs.word_ids()
    pred_labels = []
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            pred_labels.append(id_to_label[predictions[idx]])
    return list(zip(tokens, pred_labels))

def get_price(text):
    model_path = "../../models"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Example usage:
    # text = "ዋጋ 1000 ብር በአዲስ አበባ ይሸጣል።"
    result = predict_ner(text, model, tokenizer, id_to_label)
    print(result)

    b_price_tokens = [token for token, label in result if label == "B-PRICE"]
    return b_price_tokens[0] if b_price_tokens else None