"""
Quick demo without heavy models - for thesis committee to see pipeline flow
Uses dummy models but shows full SETU logic
"""

class DummySLM:
    def generate(self, prompt, max_new_tokens=512, temperature=0.7, **kwargs):
        # Simulate Bengali SLM with some hallucinations
        if "রাজধানী" in prompt:
            return "ঢাকা বাংলাদেশের রাজধানী এবং এটি বুড়িগঙ্গা নদীর তীরে অবস্থিত। এর জনসংখ্যা প্রায় ৫ কোটি। বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়।"
        elif "৫+৩" in prompt or "আম" in prompt:
            return "৫+৩=৯ কারণ ৫ এর সাথে ৩ যোগ করলে ৯ হয়। তাই মোট ৯টি আম।"
        else:
            return "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এর রাজধানী ঢাকা।"

    def generate_multiple(self, prompt, n=5, **kwargs):
        base = self.generate(prompt)
        # Simulate variations for uncertainty
        return [base, base, "ঢাকা বাংলাদেশের রাজধানী। জনসংখ্যা ২ কোটি।", base, "চট্টগ্রাম বাংলাদেশের রাজধানী।"]

    def verbalized_confidence(self, query, answer):
        return 0.8

class DummyNLI:
    def entailment_score(self, premise, hypothesis):
        # Simple heuristic
        if premise.strip() == hypothesis.strip():
            return 0.95
        if "ঢাকা" in premise and "ঢাকা" in hypothesis:
            return 0.8
        if "চট্টগ্রাম" in premise and "ঢাকা" in hypothesis:
            return 0.1
        return 0.5

    def contradiction_score(self, premise, hypothesis):
        return 1 - self.entailment_score(premise, hypothesis)

    def predict(self, premise, hypothesis):
        ent = self.entailment_score(premise, hypothesis)
        if ent > 0.7:
            return "entailment", ent
        elif ent < 0.3:
            return "contradiction", 1-ent
        else:
            return "neutral", 0.5

    def are_semantically_equivalent(self, t1, t2, threshold=0.7):
        return self.entailment_score(t1, t2) > threshold and self.entailment_score(t2, t1) > threshold

class DummyRetriever:
    def retrieve_with_fallback(self, query, top_k=5):
        if "জনসংখ্যা" in query or "৫ কোটি" in query:
            return {
                "results": [
                    {"passage": "ঢাকার জনসংখ্যা প্রায় ২ কোটি ২০ লাখ (২০২২ আদমশুমারি অনুযায়ী)।", "score": 0.9, "index": 0, "lang": "bn"},
                    {"passage": "Dhaka population is about 22 million as per 2022 census.", "score": 0.85, "index": 1, "lang": "en"}
                ],
                "fallback_used": False,
                "max_score": 0.9
            }
        else:
            return {
                "results": [{"passage": "ঢাকা বাংলাদেশের রাজধানী।", "score": 0.9, "index": 0, "lang": "bn"}],
                "fallback_used": False,
                "max_score": 0.9
            }

    def retrieve(self, query, top_k=5, lang="bn"):
        return self.retrieve_with_fallback(query, top_k)["results"]

# Build pipeline with dummies
from src.config import SETUConfig
from src.pipeline.claim_decomposer import BengaliClaimDecomposer
from src.pipeline.uncertainty_scorer import UncertaintyScorer
from src.pipeline.triage_router import TriageRouter
from src.pipeline.data_driven_corrector import DataDrivenCorrector
from src.pipeline.reasoning_corrector import ReasoningCorrector
from src.pipeline.abstention import AbstentionModule
from src.pipeline.reassembly import ReassemblyModule
from src.pipeline.setu_pipeline import SETUPipeline

config = SETUConfig()
slm = DummySLM()
nli = DummyNLI()
retriever = DummyRetriever()

decomposer = BengaliClaimDecomposer(slm_generator=slm)
uncertainty = UncertaintyScorer(slm_generator=slm, nli_model=nli, config=config)
triage = TriageRouter(config=config, slm_generator=slm, retriever=retriever)
data_corr = DataDrivenCorrector(retriever=retriever, slm_generator=slm, config=config)
reason_corr = ReasoningCorrector(slm_generator=slm, nli_model=nli, config=config)
abstention = AbstentionModule(config=config)
reassembly = ReassemblyModule(slm_generator=slm, config=config)

pipeline = SETUPipeline(slm, decomposer, uncertainty, triage, data_corr, reason_corr, abstention, reassembly, config)

# Test cases
tests = [
    "বাংলাদেশের রাজধানী কোথায় এবং এর জনসংখ্যা কত?",
    "রহিমের ৫টি আম আছে, সে আরও ৩টি কিনল। তার মোট কয়টি আম হলো?",
    "বাংলাদেশ কবে স্বাধীন হয়?"
]

for q in tests:
    print("\n" + "="*80)
    print(f"QUERY: {q}")
    print("="*80)
    res = pipeline.run(query=q, verbose=True)
    print("\n--- FINAL ANSWER ---")
    print(res["final_answer"])
    print("\n--- STATS ---")
    print(res["stats"])
