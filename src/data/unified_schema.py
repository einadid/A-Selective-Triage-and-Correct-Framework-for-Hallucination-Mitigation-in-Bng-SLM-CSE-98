"""
Unified JSONL schema for SETU
Every dataset -> {id, task_type, query, context, gold_answer, hallucinated_answer, split, language}
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
import uuid

@dataclass
class SETUInstance:
    id: str
    base_id: str  # link gold <-> hallu
    task_type: str  # qa | code_mixed_qa | summarization | reasoning
    query: str
    context: Optional[str] = None  # passage for QA/summ
    gold_answer: str = ""
    hallucinated_answer: Optional[str] = None
    split: str = "test"  # train/val/test
    language: str = "bn"  # bn | bn-en-code-mixed
    metadata: Dict = None

    def to_dict(self):
        return asdict(self)

    def to_jsonl(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

def convert_tydiqa_example(raw, split="test") -> SETUInstance:
    """Convert TyDiQA GoldP Bengali example to unified schema"""
    # raw format: {'id':..., 'question':..., 'context':..., 'answers':...}
    q = raw.get("question") or raw.get("query") or ""
    ctx = raw.get("context") or raw.get("passage") or ""
    gold = raw.get("answers", {}).get("text", [""])[0] if isinstance(raw.get("answers"), dict) else raw.get("gold_answer","")
    base_id = str(raw.get("id", uuid.uuid4()))
    return SETUInstance(
        id=f"tydiqa_{base_id}",
        base_id=base_id,
        task_type="qa",
        query=q,
        context=ctx,
        gold_answer=gold,
        split=split,
        language="bn"
    )

def convert_banglachq_example(raw, split="test") -> SETUInstance:
    # BanglaCHQ-Summ: consumer health query + summary
    return SETUInstance(
        id=f"chq_{raw.get('id', uuid.uuid4())}",
        base_id=str(raw.get('id', uuid.uuid4())),
        task_type="summarization",
        query=raw.get("query") or raw.get("question") or "",
        context=raw.get("article") or raw.get("query") or "",
        gold_answer=raw.get("summary") or raw.get("gold_answer") or "",
        split=split,
        language="bn"
    )

def convert_somadhan_example(raw, split="test") -> SETUInstance:
    # SOMADHAN: Bengali math word problems
    return SETUInstance(
        id=f"somadhan_{raw.get('id', uuid.uuid4())}",
        base_id=str(raw.get('id', uuid.uuid4())),
        task_type="reasoning",
        query=raw.get("question") or raw.get("problem") or "",
        context=raw.get("context"),
        gold_answer=raw.get("answer") or raw.get("solution") or "",
        split=split,
        language="bn"
    )

def save_jsonl(instances: List[SETUInstance], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(inst.to_jsonl() + "\n")

def load_jsonl(path: str) -> List[SETUInstance]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(SETUInstance(**d))
    return out
