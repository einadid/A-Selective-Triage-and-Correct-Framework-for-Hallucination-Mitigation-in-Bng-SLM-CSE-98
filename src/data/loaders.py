"""
Dataset loaders for SETU - downloads and converts to unified schema
"""
import os
from datasets import load_dataset
from .unified_schema import convert_tydiqa_example, convert_banglachq_example, convert_somadhan_example, save_jsonl
from tqdm import tqdm

def load_tydiqa_bn(split="validation", out_path="data/processed/tydiqa_bn.jsonl", max_samples=None):
    """
    TyDiQA-GoldP Bengali: https://huggingface.co/datasets/google-research-datasets/tydiqa
    """
    print("[Loader] Loading TyDiQA...")
    try:
        ds = load_dataset("google-research-datasets/tydiqa", "secondary_task")  # need to check
        # Actually TyDiQA GoldP is separate: use "tydiqa" builder? fallback to manual
    except Exception as e:
        print(f"HF load failed {e}, trying 'tydiqa'...")
        ds = load_dataset("tydiqa", "secondary_task")

    # Filter Bengali
    bn_data = []
    for split_name in [split]:
        if split_name in ds:
            data = ds[split_name]
            # TyDiQA has language field
            for ex in tqdm(data):
                if ex.get("id", "").startswith("bengali") or ex.get("lang") == "bengali" or "bengali" in str(ex.get("id","")).lower():
                    bn_data.append(convert_tydiqa_example(ex, split=split_name))
                # If dataset already filtered
                if "bengali" not in ds[split_name].features and len(bn_data) < 5:
                    # try to include all if secondary_task is already bn?
                    pass
            if len(bn_data) == 0:
                # If no lang filter, take all as fallback for demo
                for ex in data:
                    bn_data.append(convert_tydiqa_example(ex, split=split_name))

    if max_samples:
        bn_data = bn_data[:max_samples]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_jsonl(bn_data, out_path)
    print(f"[Loader] Saved {len(bn_data)} to {out_path}")
    return bn_data

def load_banglachq_summ(out_path="data/processed/banglachq.jsonl"):
    """
    BanglaCHQ-Summ: https://github.com/alvi-khan/BanglaCHQ-Summ
    We'll try to load from HF if available, else clone github
    """
    print("[Loader] Loading BanglaCHQ-Summ...")
    instances = []
    try:
        # Try HF mirror
        ds = load_dataset("alvi-khan/BanglaCHQ-Summ")
        for split in ds:
            for ex in ds[split]:
                instances.append(convert_banglachq_example(ex, split=split))
    except:
        print("[Loader] HF not found, please manually download from https://github.com/alvi-khan/BanglaCHQ-Summ")
        # Create dummy for pipeline test
        from .unified_schema import SETUInstance
        instances = [
            SETUInstance(
                id=f"chq_dummy_{i}",
                base_id=f"dummy_{i}",
                task_type="summarization",
                query="ডায়াবেটিসের লক্ষণ কি?",
                context="ডায়াবেটিস একটি বিপাকীয় রোগ...",
                gold_answer="ডায়াবেটিসের লক্ষণ হলো ঘন ঘন প্রস্রাব, অতিরিক্ত তৃষ্ণা...",
                split="test",
                language="bn"
            ) for i in range(5)
        ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_jsonl(instances, out_path)
    print(f"[Loader] Saved {len(instances)} to {out_path}")
    return instances

def load_somadhan(out_path="data/processed/somadhan.jsonl"):
    """
    SOMADHAN: Bengali math reasoning https://data.mendeley.com/datasets/34bs5cxk9j/1
    """
    print("[Loader] Loading SOMADHAN...")
    instances = []
    try:
        ds = load_dataset("somadhan")  # try HF
        for split in ds:
            for ex in ds[split]:
                instances.append(convert_somadhan_example(ex, split=split))
    except:
        print("[Loader] SOMADHAN not on HF, manual download required from Mendeley")
        from .unified_schema import SETUInstance
        instances = [
            SETUInstance(
                id=f"somadhan_dummy_{i}",
                base_id=f"dummy_{i}",
                task_type="reasoning",
                query="রহিমের ৫টি আম আছে, সে আরও ৩টি কিনল। তার মোট কয়টি আম হলো?",
                gold_answer="৫+৩=৮ টি আম",
                split="test",
                language="bn"
            ) for i in range(5)
        ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_jsonl(instances, out_path)
    print(f"[Loader] Saved {len(instances)} to {out_path}")
    return instances

def create_code_mixed_version(input_path="data/processed/tydiqa_bn.jsonl", out_path="data/processed/code_mixed_qa.jsonl"):
    """
    Create Bangla-English code-mixed QA from TyDiQA following BenHalluEval style
    Simple heuristic: Roman transliteration + English mix (can be improved with LLM)
    """
    from .unified_schema import load_jsonl, SETUInstance
    import random

    if not os.path.exists(input_path):
        print(f"{input_path} not found")
        return []

    instances = load_jsonl(input_path)
    mixed = []
    for inst in instances:
        # Very simple code-mixing simulation - replace some words with English
        # In real thesis, use GPT to generate code-mixed as BenHalluEval did
        q_mixed = inst.query  # TODO: apply transliteration + English insertion
        # For now keep same but tag as code-mixed
        mixed.append(SETUInstance(
            id=inst.id.replace("tydiqa", "codemix"),
            base_id=inst.base_id,
            task_type="code_mixed_qa",
            query=q_mixed,
            context=inst.context,
            gold_answer=inst.gold_answer,
            split=inst.split,
            language="bn-en-code-mixed"
        ))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_jsonl(mixed, out_path)
    print(f"[Loader] Saved {len(mixed)} code-mixed to {out_path}")
    return mixed

if __name__ == "__main__":
    load_tydiqa_bn(max_samples=100)
    load_banglachq_summ()
    load_somadhan()
    create_code_mixed_version()
