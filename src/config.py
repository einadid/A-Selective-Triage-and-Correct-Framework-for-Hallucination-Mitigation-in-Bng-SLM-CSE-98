"""
SETU Config - Central configuration for the thesis
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # default small, can switch to 3B
    # Alternatives:
    # "google/gemma-2-2b-it"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # Bangla-centric: "csebuetnlp/TigerLLM-1B" if available
    use_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class RetrievalConfig:
    bn_wiki_path: str = "data/raw/bnwiki"
    en_wiki_path: str = "data/raw/enwiki"
    index_path_bn: str = "data/index/bn_faiss.index"
    index_path_en: str = "data/index/en_faiss.index"
    embedding_model: str = "BAAI/bge-m3"  # multilingual, good for bn
    # fallback: "intfloat/multilingual-e5-large"
    chunk_size: int = 256
    chunk_overlap: int = 32
    top_k: int = 5
    bn_threshold: float = 0.35  # if retrieval score < threshold -> cross-lingual fallback

@dataclass
class UncertaintyConfig:
    n_samples: int = 7  # for semantic entropy & self-consistency
    entropy_threshold: float = 0.6
    consistency_threshold: float = 0.6
    confidence_threshold: float = 0.7
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # multilingual NLI

@dataclass
class TriageConfig:
    # Rule-based + light ML classifier
    classifier_model: str = "l3cube-pune/bengali-bert"  # or use multilingual bert
    data_keywords: List[str] = field(default_factory=lambda: [
        "কবে", "কোথায়", "কে", "কত", "সাল", "তারিখ", "রাজধানী", "জনসংখ্যা",
        "capital", "date", "year", "population", "who", "when", "where"
    ])
    reasoning_keywords: List[str] = field(default_factory=lambda: [
        "কেন", "কিভাবে", "কারণ", "যদি", "তাহলে", "অতএব", "সুতরাং",
        "why", "how", "if", "therefore", "hence", "calculate", "step"
    ])

@dataclass
class EvaluationConfig:
    tasks: List[str] = field(default_factory=lambda: ["qa", "code_mixed_qa", "summarization", "reasoning"])
    primary_metric: str = "benhallu_score"
    abstention_phrase: str = "আমি নিশ্চিত নই"

@dataclass
class SETUConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    device: str = "cuda"  # auto-detect in code
