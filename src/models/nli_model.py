"""
NLI Model - for claim verification, semantic clustering, belief consistency
Uses multilingual NLI model: mDeBERTa-v3-base-mnli-xnli
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import List, Tuple

class MultilingualNLI:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NLI] Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Label mapping
        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        # Some models have different order, check config
        try:
            id2label = self.model.config.id2label
            # Normalize
            self.id2label = {int(k): v.lower() for k,v in id2label.items()}
        except:
            self.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def _get_scores(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
        # Map probs to labels
        result = {}
        for idx, prob in enumerate(probs):
            label = self.id2label.get(idx, str(idx)).lower()
            result[label] = prob.item()
        return result

    def predict(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        scores = self._get_scores(premise, hypothesis)
        # Find best
        best_label = max(scores, key=scores.get)
        # Normalize to entailment/contradiction/neutral
        # Handle variations: "entailment" vs "entail"
        if "entail" in best_label:
            norm_label = "entailment"
        elif "contra" in best_label:
            norm_label = "contradiction"
        else:
            norm_label = "neutral"
        return norm_label, scores[best_label]

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        scores = self._get_scores(premise, hypothesis)
        # Sum any entailment-like keys
        for k,v in scores.items():
            if "entail" in k:
                return v
        return scores.get("entailment", 0.0)

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        scores = self._get_scores(premise, hypothesis)
        for k,v in scores.items():
            if "contra" in k:
                return v
        return scores.get("contradiction", 0.0)

    def are_semantically_equivalent(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """For semantic entropy clustering"""
        # Two-way entailment
        s1 = self.entailment_score(text1, text2)
        s2 = self.entailment_score(text2, text1)
        return (s1 > threshold and s2 > threshold)

    def batch_entailment(self, premises: List[str], hypothesis: str) -> List[float]:
        return [self.entailment_score(p, hypothesis) for p in premises]

if __name__ == "__main__":
    nli = MultilingualNLI()
    print(nli.predict("ঢাকা বাংলাদেশের রাজধানী", "বাংলাদেশের রাজধানী ঢাকা"))
    print(nli.predict("ঢাকা বাংলাদেশের রাজধানী", "চট্টগ্রাম বাংলাদেশের রাজধানী"))
