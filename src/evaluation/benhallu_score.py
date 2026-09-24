"""
BenHalluScore - Dual-Track evaluation from BenHalluEval (arXiv 2605.31483)
Track A: on ground-truth answers to catch false alarms
Track B: on hallucinated candidates to catch misses
BenHalluScore = 0.5 * (Track A error + Track B error) - lower is better
"""
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class BenHalluScore:
    def __init__(self, nli_model=None):
        self.nli = nli_model

    def evaluate_claim(self, gold_claim: str, predicted_claim: str, is_hallucinated_candidate: bool = False) -> Dict:
        """
        Evaluate single claim
        For Track A (gold): model should NOT flag correct claims as hallucination
        For Track B (hallucinated): model should flag hallucinated claims
        """
        # Use NLI to check if predicted is same as gold
        # If gold and pred entail each other -> correct
        # If abstained -> check if appropriate

        abstention_phrases = ["আমি নিশ্চিত নই", "I am not sure", "নিশ্চিত নই", "জানি না"]

        is_abstained = any(phrase in predicted_claim for phrase in abstention_phrases)

        if self.nli:
            try:
                entail_score = self.nli.entailment_score(gold_claim, predicted_claim)
                contra_score = self.nli.contradiction_score(gold_claim, predicted_claim)
            except:
                entail_score = 0.5
                contra_score = 0.5
        else:
            # Simple string match fallback
            entail_score = 1.0 if gold_claim.strip() == predicted_claim.strip() else 0.0
            contra_score = 0.0

        return {
            "gold": gold_claim,
            "pred": predicted_claim,
            "is_abstained": is_abstained,
            "entailment": entail_score,
            "contradiction": contra_score,
            "is_hallucinated_candidate": is_hallucinated_candidate
        }

    def compute_track_scores(self, track_results: List[Dict]) -> Dict:
        """
        track_results: list of evaluation dicts for a track
        """
        if not track_results:
            return {"error_rate": 0.0, "accuracy": 1.0, "abstention_rate": 0.0}

        n = len(track_results)
        errors = 0
        abstentions = 0

        for res in track_results:
            if res["is_hallucinated_candidate"]:
                # Track B: should detect hallucination (should NOT entail gold, or should abstain/correct)
                # If model still entails hallucinated claim, it's a miss -> error
                # Actually for Track B, gold is correct, pred is from hallucinated input
                # We want model to NOT produce hallucinated content
                # So error if pred contradicts gold or is still hallucinated
                # Simplified: if contradiction high or entailment low and not abstained -> might be still wrong?
                # Let's define: For Track B, error if model fails to correct (i.e., still hallucinates)
                # We'll use: if not abstained and contradiction > 0.5 -> error (still hallucinating)
                # This is simplified, real BenHalluEval uses human labels
                if not res["is_abstained"] and res["contradiction"] > 0.6:
                    errors += 1
                # If abstained, not error (good, it avoided hallucination)
            else:
                # Track A: gold input, should NOT abstain or hallucinate
                # Error if abstained unnecessarily (over-abstention) or contradicts gold
                if res["is_abstained"]:
                    # Over-abstention -> error (false alarm)
                    errors += 1
                elif res["contradiction"] > 0.6:
                    errors += 1

            if res["is_abstained"]:
                abstentions += 1

        error_rate = errors / n if n > 0 else 0
        abstention_rate = abstentions / n if n > 0 else 0
        accuracy = 1 - error_rate

        return {
            "error_rate": error_rate,
            "accuracy": accuracy,
            "abstention_rate": abstention_rate,
            "n": n,
            "errors": errors
        }

    def compute_benhallu_score(self, track_a_results: List[Dict], track_b_results: List[Dict]) -> Dict:
        """
        Primary metric: BenHalluScore = 0.5*(Track A error + Track B error)
        """
        track_a = self.compute_track_scores(track_a_results)
        track_b = self.compute_track_scores(track_b_results)

        benhallu_score = 0.5 * (track_a["error_rate"] + track_b["error_rate"])

        return {
            "benhallu_score": benhallu_score,
            "track_a": track_a,
            "track_b": track_b,
            "interpretation": "Lower is better. 0 = perfect, 1 = worst"
        }

    def evaluate_task(self, gold_answers: List[str], predicted_answers: List[str],
                      hallucinated_answers: List[str], predicted_from_hallu: List[str]) -> Dict:
        """
        Evaluate a full task
        gold_answers: ground truth answers (Track A)
        predicted_answers: model outputs on gold inputs
        hallucinated_answers: hallucinated candidates (for reference)
        predicted_from_hallu: model outputs when given hallucinated inputs (Track B)
        """
        track_a_results = []
        for gold, pred in zip(gold_answers, predicted_answers):
            track_a_results.append(self.evaluate_claim(gold, pred, is_hallucinated_candidate=False))

        track_b_results = []
        for gold, pred_hallu in zip(gold_answers, predicted_from_hallu):
            # For Track B, we compare gold vs pred that was conditioned on hallu
            track_b_results.append(self.evaluate_claim(gold, pred_hallu, is_hallucinated_candidate=True))

        return self.compute_benhallu_score(track_a_results, track_b_results)
