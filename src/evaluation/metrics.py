"""
Additional metrics for SETU thesis
"""
from typing import List
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from rouge_score import rouge_scorer

def compute_task_accuracy(gold: List[str], pred: List[str], task_type: str = "qa") -> float:
    """Simple accuracy / F1 for QA"""
    if task_type in ["qa", "code_mixed_qa"]:
        # Exact match or F1
        correct = sum(1 for g,p in zip(gold, pred) if g.strip().lower() in p.strip().lower() or p.strip().lower() in g.strip().lower())
        return correct / len(gold) if gold else 0
    elif task_type == "summarization":
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = []
        for g,p in zip(gold, pred):
            try:
                s = scorer.score(g, p)['rougeL'].fmeasure
                scores.append(s)
            except:
                scores.append(0)
        return float(np.mean(scores)) if scores else 0
    elif task_type == "reasoning":
        # Check if final number matches
        correct = 0
        for g,p in zip(gold, pred):
            # Extract numbers
            import re
            gold_nums = re.findall(r'\d+', g)
            pred_nums = re.findall(r'\d+', p)
            if gold_nums and pred_nums and gold_nums[-1] == pred_nums[-1]:
                correct += 1
        return correct / len(gold) if gold else 0
    return 0

def abstention_metrics(results: List[dict]) -> dict:
    """Compute abstention rate and over-abstention"""
    total = len(results)
    if total == 0:
        return {"abstention_rate": 0, "over_abstention_rate": 0}

    abstained = sum(1 for r in results if r.get("abstained", False))
    # Over-abstention: abstained on gold that was correct
    over_abstained = 0
    for r in results:
        if r.get("abstained") and r.get("track") == "A":
            over_abstained += 1

    return {
        "abstention_rate": abstained / total,
        "over_abstention_rate": over_abstained / total,
        "n_abstained": abstained,
        "n_over_abstained": over_abstained
    }

def retrieval_recall_at_k(retrieved: List[List[str]], gold_evidence: List[str], k: int = 5) -> float:
    """Recall@k for retrieval"""
    hits = 0
    for ret_list, gold in zip(retrieved, gold_evidence):
        # Check if gold evidence in top-k retrieved
        if any(gold.lower() in r.lower() or r.lower() in gold.lower() for r in ret_list[:k]):
            hits += 1
    return hits / len(gold_evidence) if gold_evidence else 0
