"""
Full Benchmark Runner - Week 2
Runs all baselines + SETU pipeline on BenHalluEval and computes BenHalluScore table

Baselines (from paper):
- Raw SLM (no mitigation)
- Chain-of-Thought (CoT)
- Uniform RAG (retrieve-then-answer, no triage)
- Chain-of-Verification (CoVe)

SETU (ours):
- Selective Triage + Targeted Correction + Abstention

Metrics:
- BenHalluScore = 0.5*(Track A error + Track B error) - lower better
- Track A: false alarm on gold
- Track B: miss on hallucinated
- Also: accuracy, abstention rate, over-abstention

Usage:
  from src.evaluation.run_benchmark import BenchmarkRunner
  runner = BenchmarkRunner(slm, retriever, nli, config)
  table = runner.run_full_benchmark(instances, max_samples_per_task=50)
"""

import os
import json
import time
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from ..data.unified_schema import SETUInstance
from .benhallu_score import BenHalluScore
from .baselines import BaselineRunner


class BenchmarkRunner:
    def __init__(self, slm_generator, retriever, nli_model, config=None, setu_pipeline=None):
        self.slm = slm_generator
        self.retriever = retriever
        self.nli = nli_model
        self.config = config
        self.setu_pipeline = setu_pipeline  # SETUPipeline instance, if available
        self.baseline_runner = BaselineRunner(slm_generator, retriever, nli_model)
        self.benhallu_scorer = BenHalluScore(nli_model=nli_model)

    def _evaluate_single_method(self, method_name: str, method_fn, instances: List[SETUInstance], max_instances: int = None) -> Dict:
        """
        Evaluate a single method (baseline or SETU) on instances
        Returns predictions for Track A and Track B
        """
        if max_instances and len(instances) > max_instances:
            instances = instances[:max_instances]

        track_a_gold = []
        track_a_pred = []
        track_b_gold = []
        track_b_pred = []
        all_results = []

        print(f"\n[Benchmark] Running {method_name} on {len(instances)} instances...")

        for inst in tqdm(instances, desc=f"{method_name}"):
            try:
                # For Track A: query with gold context, model should produce correct answer
                # For Track B: we have hallucinated candidate, we want model to NOT reproduce hallucination

                # Generate prediction
                if "SETU" in method_name and self.setu_pipeline:
                    # Use full SETU pipeline
                    out = self.setu_pipeline.run(query=inst.query, context=inst.context, task_type=inst.task_type, language=inst.language, verbose=False)
                    pred = out["final_answer"]
                else:
                    # Baseline methods
                    if method_name.startswith("raw_slm"):
                        pred = self.baseline_runner.raw_slm(inst.query, inst.context)
                    elif method_name.startswith("cot"):
                        pred = self.baseline_runner.chain_of_thought(inst.query, inst.context)
                    elif method_name.startswith("uniform_rag"):
                        pred = self.baseline_runner.uniform_rag(inst.query, inst.context)
                    elif method_name.startswith("cove"):
                        pred = self.baseline_runner.chain_of_verification(inst.query, inst.context)
                    else:
                        if method_fn is not None:
                            pred = method_fn(inst.query, inst.context)
                        else:
                            pred = self.baseline_runner.raw_slm(inst.query, inst.context)

                # For BenHalluScore, we need to evaluate pred against gold
                # Track A: if instance is gold-only (no hallucinated_answer) -> Track A
                # Track B: if instance has hallucinated_answer -> we evaluate pred vs gold, but pred was generated from query that had hallucinated context? Actually in BenHalluEval, Track B is model output when given hallucinated candidate as input? Simplified: we treat all instances as both tracks based on label
                # Here: we have separate gold and hallucinated versions in instances list
                # So we need to know if this instance is gold or hallucinated

                is_hallu = inst.hallucinated_answer is not None and inst.metadata.get("hallucination_type") != "gold"
                if is_hallu:
                    # Track B: gold is ground truth, pred is model output on hallucinated query
                    # We want model to correct hallucination, so pred should entail gold, not hallucinated
                    track_b_gold.append(inst.gold_answer)
                    track_b_pred.append(pred)
                else:
                    # Track A: gold
                    track_a_gold.append(inst.gold_answer)
                    track_a_pred.append(pred)

                all_results.append({
                    "id": inst.id,
                    "task_type": inst.task_type,
                    "query": inst.query,
                    "gold": inst.gold_answer,
                    "hallucinated": inst.hallucinated_answer,
                    "pred": pred,
                    "is_hallu": is_hallu,
                    "language": inst.language
                })

            except Exception as e:
                print(f"[Benchmark] {method_name} failed on {inst.id}: {e}")
                # Count as error
                if inst.hallucinated_answer and inst.metadata.get("hallucination_type") != "gold":
                    track_b_gold.append(inst.gold_answer)
                    track_b_pred.append("")  # empty = fail
                else:
                    track_a_gold.append(inst.gold_answer)
                    track_a_pred.append("")

        # Compute BenHalluScore
        # For this we need to use BenHalluScore.evaluate_claim for each pair
        track_a_results = []
        for gold, pred in zip(track_a_gold, track_a_pred):
            track_a_results.append(self.benhallu_scorer.evaluate_claim(gold, pred, is_hallucinated_candidate=False))

        track_b_results = []
        for gold, pred in zip(track_b_gold, track_b_pred):
            track_b_results.append(self.benhallu_scorer.evaluate_claim(gold, pred, is_hallucinated_candidate=True))

        benhallu = self.benhallu_scorer.compute_benhallu_score(track_a_results, track_b_results)

        return {
            "method": method_name,
            "benhallu_score": benhallu["benhallu_score"],
            "track_a": benhallu["track_a"],
            "track_b": benhallu["track_b"],
            "track_a_gold": track_a_gold,
            "track_a_pred": track_a_pred,
            "track_b_gold": track_b_gold,
            "track_b_pred": track_b_pred,
            "all_results": all_results,
            "n_track_a": len(track_a_gold),
            "n_track_b": len(track_b_gold)
        }

    def run_full_benchmark(self, instances: List[SETUInstance], max_samples_per_method: int = 100, save_path: str = "results/benchmark_results.json") -> Dict:
        """
        Run all baselines + SETU
        """
        # Group by task for per-task breakdown
        by_task = defaultdict(list)
        for inst in instances:
            by_task[inst.task_type].append(inst)

        all_methods = ["raw_slm", "cot", "uniform_rag", "cove", "SETU"]
        overall_results = {}

        # Overall (all tasks combined)
        print(f"\n{'='*80}")
        print(f"[Benchmark] Running full benchmark on {len(instances)} instances")
        print(f"Tasks: {list(by_task.keys())}")
        print(f"Methods: {all_methods}")
        print(f"{'='*80}")

        for method in all_methods:
            if method == "SETU" and self.setu_pipeline is None:
                print(f"[Benchmark] Skipping SETU - pipeline not provided")
                continue

            method_fn = None
            if method == "raw_slm":
                method_fn = self.baseline_runner.raw_slm
            elif method == "cot":
                method_fn = self.baseline_runner.chain_of_thought
            elif method == "uniform_rag":
                method_fn = self.baseline_runner.uniform_rag
            elif method == "cove":
                method_fn = self.baseline_runner.chain_of_verification
            elif method == "SETU":
                method_fn = None  # handled specially

            result = self._evaluate_single_method(method, method_fn, instances, max_instances=max_samples_per_method)
            overall_results[method] = result

        # Per-task breakdown
        per_task_results = {}
        for task, task_instances in by_task.items():
            print(f"\n[Benchmark] Per-task: {task} ({len(task_instances)} instances)")
            per_task_results[task] = {}
            for method in all_methods:
                if method == "SETU" and self.setu_pipeline is None:
                    continue
                method_fn = None
                if method == "raw_slm":
                    method_fn = self.baseline_runner.raw_slm
                elif method == "cot":
                    method_fn = self.baseline_runner.chain_of_thought
                elif method == "uniform_rag":
                    method_fn = self.baseline_runner.uniform_rag
                elif method == "cove":
                    method_fn = self.baseline_runner.chain_of_verification

                # Limit per task
                task_max = max_samples_per_method // len(by_task) if max_samples_per_method else None
                res = self._evaluate_single_method(f"{method}_{task}", method_fn, task_instances, max_instances=task_max)
                per_task_results[task][method] = {
                    "benhallu_score": res["benhallu_score"],
                    "track_a_error": res["track_a"]["error_rate"],
                    "track_b_error": res["track_b"]["error_rate"],
                    "n_a": res["n_track_a"],
                    "n_b": res["n_track_b"]
                }

        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Don't save all_results (too big) in JSON, only metrics
        save_data = {
            "overall": {m: {"benhallu_score": r["benhallu_score"], "track_a": r["track_a"], "track_b": r["track_b"], "n_a": r["n_track_a"], "n_b": r["n_track_b"]} for m, r in overall_results.items()},
            "per_task": per_task_results,
            "timestamp": time.time()
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        # Generate markdown table (Table 1 for thesis)
        table_md = self.generate_table_markdown(overall_results, per_task_results, save_path.replace(".json", ".md"))

        return {"overall": overall_results, "per_task": per_task_results, "table": table_md}

    def generate_table_markdown(self, overall_results: Dict, per_task_results: Dict, save_path: str) -> str:
        lines = []
        lines.append("# SETU Benchmark Results - BenHalluScore Table (Table 1)")
        lines.append("")
        lines.append("**Lower BenHalluScore is better (0=perfect, 1=worst)**")
        lines.append("BenHalluScore = 0.5*(Track A error + Track B error)")
        lines.append("- Track A error: false alarm on correct answers (over-flagging)")
        lines.append("- Track B error: miss on hallucinated candidates")
        lines.append("")

        # Overall table
        lines.append("## Overall (All Tasks)")
        lines.append("| Method | BenHalluScore ↓ | Track A Error ↓ | Track B Error ↓ | Track A Acc ↑ | Track B Acc ↑ |")
        lines.append("|---|---|---|---|---|---|")
        # Sort by BenHalluScore
        sorted_methods = sorted(overall_results.items(), key=lambda x: x[1]["benhallu_score"])
        for method, res in sorted_methods:
            lines.append(f"| {method} | {res['benhallu_score']:.3f} | {res['track_a']['error_rate']:.3f} | {res['track_b']['error_rate']:.3f} | {res['track_a']['accuracy']:.3f} | {res['track_b']['accuracy']:.3f} |")

        lines.append("")
        lines.append("## Per-Task Breakdown (BenHalluScore)")
        # Header
        tasks = list(per_task_results.keys())
        header = "| Method | " + " | ".join(tasks) + " | Overall |"
        lines.append(header)
        lines.append("|---" + "|---" * (len(tasks)+1) + "|")

        # Collect methods
        methods = set()
        for task_data in per_task_results.values():
            methods.update(task_data.keys())
        methods = sorted(methods)

        for method in methods:
            row = [method]
            for task in tasks:
                if method in per_task_results[task]:
                    row.append(f"{per_task_results[task][method]['benhallu_score']:.3f}")
                else:
                    row.append("-")
            # Overall
            if method in overall_results:
                row.append(f"{overall_results[method]['benhallu_score']:.3f}")
            else:
                row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Per-Task Breakdown (Track A Error - False Alarm)")
        lines.append("| Method | " + " | ".join(tasks) + " |")
        lines.append("|---" + "|---" * len(tasks) + "|")
        for method in methods:
            row = [method]
            for task in tasks:
                if method in per_task_results[task]:
                    row.append(f"{per_task_results[task][method]['track_a_error']:.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Per-Task Breakdown (Track B Error - Miss)")
        lines.append("| Method | " + " | ".join(tasks) + " |")
        lines.append("|---" + "|---" * len(tasks) + "|")
        for method in methods:
            row = [method]
            for task in tasks:
                if method in per_task_results[task]:
                    row.append(f"{per_task_results[task][method]['track_b_error']:.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Interpretation for Thesis")
        if overall_results:
            best = min(overall_results.items(), key=lambda x: x[1]["benhallu_score"])
            lines.append(f"- Best overall: **{best[0]}** with BenHalluScore {best[1]['benhallu_score']:.3f}")
            if "SETU" in overall_results:
                setu_score = overall_results["SETU"]["benhallu_score"]
                # Compare to second best baseline
                baseline_scores = [(m, r["benhallu_score"]) for m, r in overall_results.items() if m != "SETU"]
                if baseline_scores:
                    second_best = min(baseline_scores, key=lambda x: x[1])
                    improvement = second_best[1] - setu_score
                    if second_best[1] > 0:
                        pct = improvement/second_best[1]*100
                    else:
                        pct = 0.0
                    lines.append(f"- SETU vs best baseline ({second_best[0]}): {setu_score:.3f} vs {second_best[1]:.3f} (improvement {improvement:.3f}, {pct:.1f}%)")
                    if improvement > 0:
                        lines.append(f"  -> SETU outperforms baselines, thesis claim validated!")
                    else:
                        lines.append(f"  -> SETU needs improvement, check pipeline")
            lines.append(f"- CoT effect: Compare raw_slm vs cot - does CoT increase Track A error? (BenHalluEval found CoT increases hallucination in 12/21 cases)")
            # Check CoT
            if "raw_slm" in overall_results and "cot" in overall_results:
                raw_a = overall_results["raw_slm"]["track_a"]["error_rate"]
                cot_a = overall_results["cot"]["track_a"]["error_rate"]
                if cot_a > raw_a:
                    lines.append(f"  -> CoT increases false alarm: Raw Track A {raw_a:.3f} -> CoT Track A {cot_a:.3f} (matches BenHalluEval finding)")

        report = "\n".join(lines)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"\n[Benchmark] Saved table to {save_path}")
        return report


# Dummy test
if __name__ == "__main__":
    class DummySLM:
        def generate(self, prompt, max_new_tokens=200, **kwargs):
            return "ঢাকা বাংলাদেশের রাজধানী।"
    class DummyRet:
        def retrieve_with_fallback(self, q, top_k=5):
            return {"results": [{"passage": "ঢাকা রাজধানী", "score": 0.9}], "fallback_used": False, "max_score": 0.9}
        def retrieve(self, q, top_k=5, lang="bn"):
            return [{"passage": "ঢাকা রাজধানী", "score": 0.9}]
    class DummyNLI:
        def entailment_score(self, a, b):
            return 0.8
        def contradiction_score(self, a, b):
            return 0.2
        def predict(self, a, b):
            return "entailment", 0.8
        def are_semantically_equivalent(self, a, b, threshold=0.7):
            return True

    from src.config import SETUConfig
    from src.data.benhallu_eval_loader import BenHalluEvalLoader
    from src.pipeline.setu_pipeline import SETUPipeline
    from src.pipeline.claim_decomposer import BengaliClaimDecomposer
    from src.pipeline.uncertainty_scorer import UncertaintyScorer
    from src.pipeline.triage_router import TriageRouter
    from src.pipeline.data_driven_corrector import DataDrivenCorrector
    from src.pipeline.reasoning_corrector import ReasoningCorrector
    from src.pipeline.abstention import AbstentionModule
    from src.pipeline.reassembly import ReassemblyModule

    cfg = SETUConfig()
    loader = BenHalluEvalLoader()
    data = loader.load_all(max_samples_per_task=5)

    slm = DummySLM()
    ret = DummyRet()
    nli = DummyNLI()

    dec = BengaliClaimDecomposer(slm)
    unc = UncertaintyScorer(slm, nli, cfg)
    triage = TriageRouter(cfg, slm, ret)
    ddc = DataDrivenCorrector(ret, slm, cfg)
    rc = ReasoningCorrector(slm, nli, cfg)
    abst = AbstentionModule(cfg)
    reasm = ReassemblyModule(slm, cfg)
    pipe = SETUPipeline(slm, dec, unc, triage, ddc, rc, abst, reasm, cfg)

    runner = BenchmarkRunner(slm, ret, nli, cfg, setu_pipeline=pipe)
    runner.run_full_benchmark(data, max_samples_per_method=10, save_path="results/benchmark_dummy.json")
