"""
BenHalluEval 12K Loader - Week 2
Paper: BenHalluEval: Multi-Task Hallucination Evaluation for Bengali (arXiv 2605.31483)
4 tasks: GQA (TyDiQA), Code-Mixed QA, Summarization (BanglaCHQ), Reasoning (SOMADHAN)
12,000 hallucinated candidates = 4000 GQA + 4000 CodeMixed + 3000 Summ + 1000 Reasoning
+ 4000 ground truth = total 16k instances for dual-track eval.

This loader:
- Tries to load real BenHalluEval if present (anonymous.4open.science)
- If not, builds synthetic version from seed datasets (TyDiQA, BanglaCHQ, SOMADHAN)
  using heuristic hallucination generation that mimics paper's 12 types.

Usage:
  from src.data.benhallu_eval_loader import BenHalluEvalLoader
  loader = BenHalluEvalLoader(data_dir="data")
  dataset = loader.load_all(max_samples_per_task=100)  # 100 per task for quick test
"""

import os
import json
import random
import re
from typing import List, Dict, Optional
from dataclasses import asdict

from .unified_schema import SETUInstance, save_jsonl, load_jsonl

# For real HF datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False


class BenHalluEvalLoader:
    """
    Unified loader for BenHalluEval 12K
    """
    def __init__(self, data_dir: str = "data", seed: int = 42):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        random.seed(seed)

        # Hallucination type definitions from paper
        self.hallu_types = {
            "qa": ["factualness", "comprehension", "specificity", "inference"],
            "code_mixed_qa": ["factualness", "comprehension", "specificity", "inference"],
            "summarization": ["fabricated_content", "non_factual_addition", "direct_contradiction"],
            "reasoning": ["arithmetic_error", "wrong_formula", "step_omission", "hallucinated_fact", "semantic_drift"]
        }

    # ---------------- Real BenHalluEval detection ----------------
    def _try_load_real_benhallu(self, path: Optional[str] = None) -> Optional[List[SETUInstance]]:
        """
        If user manually downloaded BenHalluEval from https://anonymous.4open.science/r/BanglaHalluEval-EB77
        Place jsonl at data/raw/benhallu_eval.jsonl or data/processed/benhallu_eval.jsonl
        Expected format: each line {id, task_type, query, context, gold_answer, hallucinated_answer, hallucination_type, language, split}
        """
        candidates = [
            path,
            os.path.join(self.raw_dir, "benhallu_eval.jsonl"),
            os.path.join(self.processed_dir, "benhallu_eval.jsonl"),
            os.path.join(self.raw_dir, "BenHalluEval.jsonl"),
            "data/benhallu_eval.jsonl"
        ]
        for p in candidates:
            if p and os.path.exists(p):
                print(f"[BenHalluEvalLoader] Found real BenHalluEval at {p}")
                try:
                    instances = []
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            d = json.loads(line)
                            # Convert to SETUInstance
                            instances.append(SETUInstance(
                                id=d.get("id", f"benhallu_{len(instances)}"),
                                base_id=d.get("base_id", d.get("id", "")),
                                task_type=d.get("task_type", "qa"),
                                query=d.get("query", ""),
                                context=d.get("context"),
                                gold_answer=d.get("gold_answer", ""),
                                hallucinated_answer=d.get("hallucinated_answer"),
                                split=d.get("split", "test"),
                                language=d.get("language", "bn"),
                                metadata={"hallucination_type": d.get("hallucination_type"), "source": "real_benhallu"}
                            ))
                    print(f"[BenHalluEvalLoader] Loaded {len(instances)} real instances")
                    return instances
                except Exception as e:
                    print(f"[BenHalluEvalLoader] Failed to parse real file {p}: {e}")
        return None

    # ---------------- Seed dataset loaders ----------------
    def _load_tydiqa_seed(self, max_samples: int = 1000) -> List[Dict]:
        """Load TyDiQA GoldP Bengali as seed for GQA + CodeMixed"""
        print("[BenHalluEvalLoader] Loading TyDiQA GoldP seed...")
        seed_data = []
        if HF_AVAILABLE:
            try:
                # Try TyDiQA GoldP - there are multiple HF configs
                # Most reliable: load_dataset("tydiqa", "secondary_task") then filter bn
                # Or "google-research-datasets/tydiqa" with language filter
                ds = None
                try:
                    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task")
                except:
                    try:
                        ds = load_dataset("tydiqa", "secondary_task")
                    except:
                        ds = load_dataset("tydiqa", "primary_task")

                # TyDiQA has splits: train, validation
                # We need Bengali examples
                for split_name in ["validation", "train"]:
                    if split_name not in ds:
                        continue
                    split_data = ds[split_name]
                    for ex in split_data:
                        # Check Bengali - TyDiQA secondary_task has 'id' like 'bengali-xxx' or lang field
                        is_bn = False
                        ex_id = str(ex.get("id", "")).lower()
                        if "bengali" in ex_id or "bn" in ex_id:
                            is_bn = True
                        # Some versions have 'language' column
                        if ex.get("language") == "bengali" or ex.get("lang") == "bengali":
                            is_bn = True
                        # If dataset is already filtered, take all
                        # For secondary_task, the dataset is per language? We'll check first 5 examples
                        if len(seed_data) < 5:
                            # Heuristic: if question contains Bengali chars, it's Bengali
                            q = ex.get("question", "")
                            if any('\u0980' <= c <= '\u09FF' for c in q):
                                is_bn = True

                        if is_bn:
                            seed_data.append({
                                "id": ex.get("id", f"tydiqa_{len(seed_data)}"),
                                "question": ex.get("question", ""),
                                "context": ex.get("context", "") or ex.get("document_plaintext", "")[:2000],
                                "answer": ex.get("answers", {}).get("text", [""])[0] if isinstance(ex.get("answers"), dict) else ex.get("answer", "")[:500],
                                "language": "bn"
                            })
                            if len(seed_data) >= max_samples:
                                break
                    if len(seed_data) >= max_samples:
                        break
                print(f"[BenHalluEvalLoader] Got {len(seed_data)} TyDiQA BN examples")
            except Exception as e:
                print(f"[BenHalluEvalLoader] HF TyDiQA load failed: {e}")
        # Fallback dummy
        if len(seed_data) == 0:
            print("[BenHalluEvalLoader] Using dummy TyDiQA seed (offline)")
            dummy_qa = [
                ("বাংলাদেশের রাজধানী কোথায়?", "ঢাকা বাংলাদেশের রাজধানী।", "ঢাকা"),
                ("বাংলাদেশ কবে স্বাধীন হয়?", "বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়।", "১৯৭১"),
                ("পদ্মা সেতুর দৈর্ঘ্য কত?", "পদ্মা সেতুর দৈর্ঘ্য ৬.১৫ কিলোমিটার।", "৬.১৫ কিলোমিটার"),
                ("বাংলাদেশের জাতীয় ফুল কি?", "বাংলাদেশের জাতীয় ফুল শাপলা।", "শাপলা"),
                ("রবীন্দ্রনাথ ঠাকুর কবে নোবেল পান?", "রবীন্দ্রনাথ ১৯১৩ সালে নোবেল পান।", "১৯১৩"),
            ]
            for i, (q, ctx, ans) in enumerate(dummy_qa * (max_samples // 5 + 1)):
                if i >= max_samples:
                    break
                seed_data.append({"id": f"dummy_tydiqa_{i}", "question": q, "context": ctx, "answer": ans, "language": "bn"})
        return seed_data[:max_samples]

    def _load_banglachq_seed(self, max_samples: int = 1000) -> List[Dict]:
        print("[BenHalluEvalLoader] Loading BanglaCHQ-Summ seed...")
        seed_data = []
        # Try HF if available
        if HF_AVAILABLE:
            try:
                # Not on HF usually, try github clone? For now dummy
                pass
            except:
                pass
        if len(seed_data) == 0:
            dummy_summ = [
                ("ডায়াবেটিসের লক্ষণ কি?", "ডায়াবেটিস একটি বিপাকীয় রোগ যেখানে রক্তে শর্করার মাত্রা বেশি থাকে। লক্ষণ: ঘন ঘন প্রস্রাব, অতিরিক্ত তৃষ্ণা, ক্লান্তি।", "ডায়াবেটিসের লক্ষণ: ঘন ঘন প্রস্রাব, অতিরিক্ত তৃষ্ণা।"),
                ("উচ্চ রক্তচাপের কারণ কি?", "উচ্চ রক্তচাপের কারণ: অতিরিক্ত লবণ, স্থূলতা, ধূমপান।", "উচ্চ রক্তচাপের কারণ অতিরিক্ত লবণ ও স্থূলতা।"),
            ]
            for i, (q, ctx, summ) in enumerate(dummy_summ * (max_samples // 2 + 1)):
                if i >= max_samples:
                    break
                seed_data.append({"id": f"dummy_chq_{i}", "question": q, "context": ctx, "answer": summ, "language": "bn"})
        return seed_data[:max_samples]

    def _load_somadhan_seed(self, max_samples: int = 1000) -> List[Dict]:
        print("[BenHalluEvalLoader] Loading SOMADHAN seed...")
        seed_data = []
        dummy_math = [
            ("রহিমের ৫টি আম আছে, সে আরও ৩টি কিনল। মোট কয়টি?", "৫+৩=৮, উত্তর ৮টি আম।", "৮"),
            ("একটি বইয়ের দাম ১২০ টাকা, ৩টি বইয়ের দাম কত?", "১২০*৩=৩৬০ টাকা।", "৩৬০ টাকা"),
            ("একটি ট্রেন ঘণ্টায় ৬০ কিমি যায়, ৩ ঘণ্টায় কত যাবে?", "৬০*৩=১৮০ কিমি।", "১৮০ কিমি"),
        ]
        for i, (q, ctx, ans) in enumerate(dummy_math * (max_samples // 3 + 1)):
            if i >= max_samples:
                break
            seed_data.append({"id": f"dummy_somadhan_{i}", "question": q, "context": ctx, "answer": ans, "language": "bn"})
        return seed_data[:max_samples]

    # ---------------- Synthetic hallucination generation ----------------
    def _hallucinate_qa(self, question: str, context: str, gold: str, hallu_type: str) -> str:
        """Heuristic hallucination for QA, mimics paper's 4 types"""
        gold = gold.strip()
        if hallu_type == "factualness":
            # Entity / date / place swap
            # Number perturbation
            if re.search(r'\d+', gold):
                # Change number
                return re.sub(r'\d+', lambda m: str(int(m.group(0)) + random.randint(1, 10)), gold, count=1) + " (ভুল তথ্য)"
            else:
                # Entity swap
                swaps = {"ঢাকা": "চট্টগ্রাম", "বাংলাদেশ": "ভারত", "১৯৭১": "১৯৮০", "শাপলা": "গোলাপ"}
                for k, v in swaps.items():
                    if k in gold:
                        return gold.replace(k, v)
                return gold + " যা আসলে ভুল তথ্য।"
        elif hallu_type == "comprehension":
            return f"প্রশ্নটি ভুল বুঝে উত্তর: {question} এর উত্তর হলো এটি একটি সুন্দর জায়গা।"
        elif hallu_type == "specificity":
            # Too vague or too specific
            if random.random() < 0.5:
                return "এটি একটি স্থান।"  # too vague
            else:
                return gold + " এবং এর আশেপাশে ৫টি নদী, ১০টি স্কুল, ২০টি দোকান আছে যা খুবই নির্দিষ্ট ভুল তথ্য।"
        elif hallu_type == "inference":
            return gold + " তাই এটি থেকে প্রমাণিত হয় যে বাংলাদেশ ইউরোপে অবস্থিত।"
        return gold + " [hallucinated]"

    def _hallucinate_code_mixed(self, question_bn: str, gold: str, hallu_type: str) -> str:
        """Code-mixed version - Roman script + English mix"""
        # Simple transliteration heuristic: keep some English
        base = self._hallucinate_qa(question_bn, "", gold, hallu_type)
        # Convert to code-mixed style
        code_mixed_map = {
            "ঢাকা": "Dhaka", "বাংলাদেশ": "Bangladesh", "রাজধানী": "capital",
            "জনসংখ্যা": "population", "স্বাধীন": "independent"
        }
        cm = base
        for bn, en in code_mixed_map.items():
            cm = cm.replace(bn, f"{en} ({bn})")
        # Add Roman mix
        if random.random() < 0.5:
            cm = "Dhaka holo Bangladesh er capital, " + cm
        return cm

    def _hallucinate_summ(self, query: str, context: str, gold_summ: str, hallu_type: str) -> str:
        if hallu_type == "fabricated_content":
            return gold_summ + " এছাড়া রোগীর হার্টে একটি অতিরিক্ত ভালভ পাওয়া গেছে যা ডকুমেন্টে নেই।"
        elif hallu_type == "non_factual_addition":
            return gold_summ + " এই রোগটি সাধারণত ৫০ বছর বয়সের পর হয় এবং এর চিকিৎসা খুব ব্যয়বহুল।"
        elif hallu_type == "direct_contradiction":
            # Reverse
            if "বেশি" in gold_summ:
                return gold_summ.replace("বেশি", "কম")
            elif "ঘন ঘন" in gold_summ:
                return gold_summ.replace("ঘন ঘন", "কখনোই না")
            else:
                return "এই রোগের কোনো লক্ষণ নেই।"  # contradiction
        return gold_summ + " [hallucinated summary]"

    def _hallucinate_reasoning(self, question: str, gold_reasoning: str, hallu_type: str) -> str:
        if hallu_type == "arithmetic_error":
            # Change final number
            return re.sub(r'\d+', lambda m: str(int(m.group(0)) + 1), gold_reasoning, count=1)
        elif hallu_type == "wrong_formula":
            return gold_reasoning.replace("+", "-").replace("*", "+") + " (ভুল সূত্র ব্যবহার)"
        elif hallu_type == "step_omission":
            # Remove middle step
            steps = gold_reasoning.split("।")
            if len(steps) > 2:
                return "।".join([steps[0], steps[-1]])
            return gold_reasoning
        elif hallu_type == "hallucinated_fact":
            return gold_reasoning + " এবং পিথাগোরাসের উপপাদ্য অনুযায়ী এটি সত্য।"
        elif hallu_type == "semantic_drift":
            return "এই সমস্যাটি সমাধান করতে হলে আমাদের প্রথমে আকাশের রঙ নিয়ে ভাবতে হবে, কারণ নীল আকাশ সুন্দর।"
        return gold_reasoning + " [reasoning hallucinated]"

    def _build_synthetic_dataset(self, max_samples_per_task: int = 100) -> List[SETUInstance]:
        """
        Build synthetic BenHalluEval-like dataset
        max_samples_per_task = number of seed samples per task (each seed generates multiple hallu)
        For paper: 1000 seed per task -> 4000 hallu for QA (4 per seed)
        For quick test: 100 seed -> 400 hallu for QA
        """
        all_instances = []

        # GQA
        tydiqa_seed = self._load_tydiqa_seed(max_samples=max_samples_per_task)
        print(f"[Synthetic] Building GQA from {len(tydiqa_seed)} seeds...")
        for seed in tydiqa_seed:
            base_id = seed["id"]
            for h_type in self.hallu_types["qa"]:
                hallu_ans = self._hallucinate_qa(seed["question"], seed["context"], seed["answer"], h_type)
                inst = SETUInstance(
                    id=f"gqa_{base_id}_{h_type}",
                    base_id=base_id,
                    task_type="qa",
                    query=seed["question"],
                    context=seed["context"],
                    gold_answer=seed["answer"],
                    hallucinated_answer=hallu_ans,
                    split="test",
                    language="bn",
                    metadata={"hallucination_type": h_type, "source": "synthetic"}
                )
                all_instances.append(inst)
            # Also add gold-only for Track A
            all_instances.append(SETUInstance(
                id=f"gqa_{base_id}_gold",
                base_id=base_id,
                task_type="qa",
                query=seed["question"],
                context=seed["context"],
                gold_answer=seed["answer"],
                hallucinated_answer=None,
                split="test",
                language="bn",
                metadata={"hallucination_type": "gold", "source": "synthetic"}
            ))

        # Code-Mixed QA
        print(f"[Synthetic] Building Code-Mixed QA from {len(tydiqa_seed)} seeds...")
        for seed in tydiqa_seed:
            base_id = seed["id"]
            # Convert question to code-mixed (simple heuristic)
            q_cm = seed["question"].replace("কোথায়", "kothay").replace("কি", "ki").replace("কত", "koto")
            q_cm = f"{q_cm} (Dhaka er bepare bolo)"
            for h_type in self.hallu_types["code_mixed_qa"]:
                hallu_ans = self._hallucinate_code_mixed(seed["question"], seed["answer"], h_type)
                inst = SETUInstance(
                    id=f"codemix_{base_id}_{h_type}",
                    base_id=base_id,
                    task_type="code_mixed_qa",
                    query=q_cm,
                    context=seed["context"],
                    gold_answer=seed["answer"],
                    hallucinated_answer=hallu_ans,
                    split="test",
                    language="bn-en-code-mixed",
                    metadata={"hallucination_type": h_type, "source": "synthetic"}
                )
                all_instances.append(inst)

        # Summarization
        chq_seed = self._load_banglachq_seed(max_samples=max_samples_per_task)
        print(f"[Synthetic] Building Summarization from {len(chq_seed)} seeds...")
        for seed in chq_seed:
            base_id = seed["id"]
            for h_type in self.hallu_types["summarization"]:
                hallu_ans = self._hallucinate_summ(seed["question"], seed["context"], seed["answer"], h_type)
                inst = SETUInstance(
                    id=f"summ_{base_id}_{h_type}",
                    base_id=base_id,
                    task_type="summarization",
                    query=seed["question"],
                    context=seed["context"],
                    gold_answer=seed["answer"],
                    hallucinated_answer=hallu_ans,
                    split="test",
                    language="bn",
                    metadata={"hallucination_type": h_type, "source": "synthetic"}
                )
                all_instances.append(inst)

        # Reasoning
        somadhan_seed = self._load_somadhan_seed(max_samples=max_samples_per_task)
        print(f"[Synthetic] Building Reasoning from {len(somadhan_seed)} seeds...")
        # For reasoning, 5 types but only 200 per type in paper (1000 total). For synthetic, distribute
        for idx, seed in enumerate(somadhan_seed):
            base_id = seed["id"]
            h_type = self.hallu_types["reasoning"][idx % len(self.hallu_types["reasoning"])]
            hallu_ans = self._hallucinate_reasoning(seed["question"], seed["answer"], h_type)
            inst = SETUInstance(
                id=f"reasoning_{base_id}_{h_type}",
                base_id=base_id,
                task_type="reasoning",
                query=seed["question"],
                context=seed["context"],
                gold_answer=seed["answer"],
                hallucinated_answer=hallu_ans,
                split="test",
                language="bn",
                metadata={"hallucination_type": h_type, "source": "synthetic"}
            )
            all_instances.append(inst)

        print(f"[Synthetic] Total instances built: {len(all_instances)}")
        return all_instances

    # ---------------- Public API ----------------
    def load_all(self, max_samples_per_task: int = 100, use_real_if_available: bool = True, save_path: Optional[str] = None) -> List[SETUInstance]:
        """
        Main entry: load BenHalluEval dataset
        max_samples_per_task: for synthetic mode, how many seed samples per task
        """
        if use_real_if_available:
            real = self._try_load_real_benhallu()
            if real is not None:
                return real

        # Build synthetic
        synthetic = self._build_synthetic_dataset(max_samples_per_task=max_samples_per_task)

        # Save
        if save_path is None:
            save_path = os.path.join(self.processed_dir, f"benhallu_eval_synthetic_{max_samples_per_task}per_task.jsonl")
        save_jsonl(synthetic, save_path)
        print(f"[BenHalluEvalLoader] Saved synthetic dataset to {save_path}")
        return synthetic

    def get_dual_track_split(self, instances: List[SETUInstance]):
        """
        Split into Track A (gold) and Track B (hallucinated) for BenHalluScore
        Returns dict task_type -> {track_a: [...], track_b: [...]}
        """
        by_task = {}
        for inst in instances:
            task = inst.task_type
            if task not in by_task:
                by_task[task] = {"track_a": [], "track_b": []}
            if inst.hallucinated_answer is None or inst.metadata.get("hallucination_type") == "gold":
                by_task[task]["track_a"].append(inst)
            else:
                by_task[task]["track_b"].append(inst)
        return by_task

    def stats(self, instances: List[SETUInstance]):
        """Print stats"""
        from collections import Counter
        print(f"\n[BenHalluEval Stats] Total: {len(instances)}")
        task_counts = Counter([i.task_type for i in instances])
        print(f"By task: {task_counts}")
        lang_counts = Counter([i.language for i in instances])
        print(f"By language: {lang_counts}")
        hallu_counts = Counter([i.metadata.get('hallucination_type') for i in instances if i.metadata])
        print(f"By hallucination type: {hallu_counts}")
        print()


# CLI test
if __name__ == "__main__":
    loader = BenHalluEvalLoader()
    data = loader.load_all(max_samples_per_task=10)
    loader.stats(data)
    split = loader.get_dual_track_split(data)
    for task, tracks in split.items():
        print(f"Task {task}: Track A {len(tracks['track_a'])}, Track B {len(tracks['track_b'])}")
