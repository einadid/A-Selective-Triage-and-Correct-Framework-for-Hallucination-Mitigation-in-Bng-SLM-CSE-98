"""
Uncertainty Scoring - RQ1: Which signal is calibrated for Bengali SLMs?
Three signals: Semantic Entropy, Self-Consistency, Verbalized Confidence
"""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
import math

class UncertaintyScorer:
    def __init__(self, slm_generator, nli_model, config=None):
        self.slm = slm_generator
        self.nli = nli_model
        self.config = config
        self.n_samples = config.uncertainty.n_samples if config else 7

    def semantic_entropy(self, query: str, claim: str, samples: List[str] = None) -> float:
        """
        Farquhar et al., Nature 2024
        Cluster samples by meaning via NLI, then compute entropy over clusters
        Higher entropy = more uncertainty = more likely hallucination
        """
        if samples is None:
            # Generate multiple answers for the claim context
            prompt = f"প্রশ্ন: {query}\nএই দাবিটি যাচাই করো: {claim}\nউত্তর দাও:"
            samples = self.slm.generate_multiple(prompt, n=self.n_samples, max_new_tokens=128)

        # Cluster by semantic equivalence
        clusters = []  # list of list of indices
        cluster_assignments = []

        for i, s in enumerate(samples):
            assigned = False
            for c_idx, cluster in enumerate(clusters):
                # Compare with representative of cluster
                rep = samples[cluster[0]]
                if self.nli.are_semantically_equivalent(s, rep, threshold=0.7):
                    cluster.append(i)
                    cluster_assignments.append(c_idx)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
                cluster_assignments.append(len(clusters)-1)

        # Compute entropy
        if len(clusters) <= 1:
            return 0.0  # all same meaning -> low uncertainty

        # Probability of each cluster
        total = len(samples)
        probs = [len(c)/total for c in clusters]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        # Normalize by log(num_samples) to get [0,1]
        max_entropy = math.log(total) if total > 1 else 1
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return norm_entropy

    def self_consistency(self, claim: str, samples: List[str] = None, query: str = "") -> float:
        """
        SelfCheckGPT (Manakul et al., EMNLP 2023)
        Agreement across samples - lower agreement = higher uncertainty
        Returns consistency score [0,1] - higher means more consistent (less hallucination)
        """
        if samples is None:
            prompt = f"প্রশ্ন: {query}\nদাবি: {claim}\nএটি সত্য কিনা যাচাই করো:"
            samples = self.slm.generate_multiple(prompt, n=self.n_samples, max_new_tokens=128)

        # Compute entailment of claim by each sample
        scores = []
        for s in samples:
            # Does sample entail claim?
            try:
                score = self.nli.entailment_score(s, claim)
                scores.append(score)
            except:
                scores.append(0.5)

        if not scores:
            return 0.5

        # Consistency = mean entailment
        consistency = float(np.mean(scores))
        return consistency

    def verbalized_confidence(self, query: str, claim: str) -> float:
        """Direct confidence from model"""
        try:
            return self.slm.verbalized_confidence(query, claim)
        except:
            return 0.5

    def score_claim(self, query: str, claim: str, samples: List[str] = None) -> Dict[str, float]:
        """
        Compute all three signals for a claim
        Returns dict with scores and final flag
        """
        if samples is None:
            prompt = f"প্রশ্ন: {query}\nউত্তর/দাবি: {claim}\nএই সম্পর্কে বিস্তারিত বলো:"
            samples = self.slm.generate_multiple(prompt, n=self.n_samples, max_new_tokens=128)

        sem_ent = self.semantic_entropy(query, claim, samples)
        self_cons = self.self_consistency(claim, samples, query)
        verb_conf = self.verbalized_confidence(query, claim)

        # Convert to uncertainty (higher = more likely hallucination)
        # Semantic entropy already uncertainty (0=certain, 1=uncertain)
        # Self-consistency is certainty, so uncertainty = 1 - consistency
        # Verbalized confidence is certainty, so uncertainty = 1 - confidence
        uncertainties = {
            "semantic_entropy": sem_ent,
            "self_consistency": 1.0 - self_cons,
            "verbalized_confidence": 1.0 - verb_conf,
            "consistency_raw": self_cons,
            "confidence_raw": verb_conf
        }

        # Combined uncertainty (simple average for now, can learn weights in calibration study)
        combined = (sem_ent + (1-self_cons) + (1-verb_conf)) / 3.0
        uncertainties["combined"] = combined

        # Flag if high uncertainty
        threshold = self.config.uncertainty.entropy_threshold if self.config else 0.6
        uncertainties["is_flagged"] = combined > threshold

        return uncertainties

    def calibration_study(self, claims: List[str], labels: List[int], scores: List[float]) -> Dict[str, float]:
        """
        RQ1: Compute ECE and AUROC for a given uncertainty signal
        labels: 1 = hallucination, 0 = correct
        scores: uncertainty scores (higher = more likely hallucination)
        """
        # AUROC
        try:
            auroc = roc_auc_score(labels, scores)
        except:
            auroc = 0.5

        # ECE - Expected Calibration Error
        # Bin predictions
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins+1)
        ece = 0.0
        total = len(labels)

        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i+1]
            mask = (np.array(scores) >= low) & (np.array(scores) < high)
            if np.sum(mask) == 0:
                continue
            bin_acc = np.mean(np.array(labels)[mask])  # actual hallucination rate in bin
            bin_conf = np.mean(np.array(scores)[mask])  # predicted uncertainty in bin
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask) / total

        return {"auroc": auroc, "ece": ece, "n": total}

    def compare_signals(self, dataset) -> Dict:
        """
        Full RQ1 study: compare 3 signals on native vs code-mixed
        dataset: list of dicts with query, claim, label, language
        """
        results = {
            "native": {"semantic_entropy": [], "self_consistency": [], "verbalized_confidence": []},
            "code_mixed": {"semantic_entropy": [], "self_consistency": [], "verbalized_confidence": []}
        }
        # Implementation would iterate dataset and collect scores
        # Placeholder for thesis - actual run on Kaggle
        return results
