"""
Main entry point for SETU
"""
import argparse
from config import SETUConfig
from models.slm_generator import SLMGenerator
from models.nli_model import MultilingualNLI
from pipeline.claim_decomposer import BengaliClaimDecomposer
from pipeline.uncertainty_scorer import UncertaintyScorer
from pipeline.triage_router import TriageRouter
from pipeline.data_driven_corrector import DataDrivenCorrector
from pipeline.reasoning_corrector import ReasoningCorrector
from pipeline.abstention import AbstentionModule
from pipeline.reassembly import ReassemblyModule
from pipeline.setu_pipeline import SETUPipeline
from retrieval.retriever import BengaliRetriever
from evaluation.benhallu_score import BenHalluScore

def build_pipeline(config: SETUConfig):
    print("[SETU] Building pipeline...")

    # Models
    slm = SLMGenerator(model_name=config.model.name, use_4bit=config.model.use_4bit)
    nli = MultilingualNLI(model_name=config.uncertainty.nli_model)

    # Retrieval
    try:
        retriever = BengaliRetriever(config)
    except Exception as e:
        print(f"[SETU] Retriever init failed: {e}, using dummy")
        retriever = None
        # Create dummy retriever for testing
        class DummyRetriever:
            def retrieve_with_fallback(self, query, top_k=5):
                return {"results": [{"passage": "ঢাকা বাংলাদেশের রাজধানী।", "score": 0.8, "index": 0, "lang": "bn"}], "fallback_used": False, "max_score": 0.8}
            def retrieve(self, query, top_k=5, lang="bn"):
                return [{"passage": "ঢাকা বাংলাদেশের রাজধানী।", "score": 0.8, "index": 0, "lang": "bn"}]
        retriever = DummyRetriever()

    # Pipeline components
    decomposer = BengaliClaimDecomposer(slm_generator=slm)
    uncertainty = UncertaintyScorer(slm_generator=slm, nli_model=nli, config=config)
    triage = TriageRouter(config=config, slm_generator=slm, retriever=retriever)
    data_corrector = DataDrivenCorrector(retriever=retriever, slm_generator=slm, config=config)
    reasoning_corrector = ReasoningCorrector(slm_generator=slm, nli_model=nli, config=config)
    abstention = AbstentionModule(config=config)
    reassembly = ReassemblyModule(slm_generator=slm, config=config)

    pipeline = SETUPipeline(
        slm_generator=slm,
        claim_decomposer=decomposer,
        uncertainty_scorer=uncertainty,
        triage_router=triage,
        data_corrector=data_corrector,
        reasoning_corrector=reasoning_corrector,
        abstention_module=abstention,
        reassembly_module=reassembly,
        config=config
    )

    return pipeline

def main():
    parser = argparse.ArgumentParser(description="SETU Framework")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="SLM model name")
    parser.add_argument("--task", type=str, default="qa", choices=["qa", "code_mixed_qa", "summarization", "reasoning"])
    parser.add_argument("--input", type=str, default="বাংলাদেশের রাজধানী কোথায়?", help="Input query")
    parser.add_argument("--context", type=str, default=None, help="Optional context")
    args = parser.parse_args()

    config = SETUConfig()
    config.model.name = args.model

    pipeline = build_pipeline(config)

    result = pipeline.run(query=args.input, context=args.context, task_type=args.task, language="bn", verbose=True)

    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Query: {result['query']}")
    print(f"Draft: {result['draft_answer']}")
    print(f"Final: {result['final_answer']}")
    print(f"Stats: {result['stats']}")
    print(f"Time: {result['time']:.2f}s")

if __name__ == "__main__":
    main()
