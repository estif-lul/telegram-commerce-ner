import torch
import numpy as np

class NERWrapper:
    """A wrapper for Named Entity Recognition (NER) models that predicts probabilities for input texts."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict_proba(self, texts):
        """Predicts probabilities for the input texts using the NER model.
        Args:
            texts (list of str): List of input texts to predict.
        Returns:
            np.ndarray: Array of predicted probabilities for each text.
        """
        outputs = []
        for text in texts:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**tokens).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            outputs.append(probs[0].mean(dim=0).numpy())  # Simplified
        return np.array(outputs)