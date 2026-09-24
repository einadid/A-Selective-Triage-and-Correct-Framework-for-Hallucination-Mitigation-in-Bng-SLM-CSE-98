"""
Data-driven Corrector - Cross-lingual RAG for factual errors
"""
from typing import Dict, List

class DataDrivenCorrector:
    def __init__(self, retriever, slm_generator, config=None):
        self.retriever = retriever
        self.slm = slm_generator
        self.config = config

    def correct_claim(self, query: str, claim: str) -> Dict:
        """
        Correct a data-driven hallucinated claim using retrieval
        """
        # Step 1: Retrieve evidence
        retrieval_out = self.retriever.retrieve_with_fallback(claim, top_k=self.config.retrieval.top_k if self.config else 5)
        evidence_list = retrieval_out["results"]
        fallback_used = retrieval_out["fallback_used"]

        if not evidence_list:
            return {
                "original_claim": claim,
                "corrected_claim": claim,
                "status": "no_evidence",
                "evidence": [],
                "fallback_used": fallback_used
            }

        # Build evidence context
        evidence_text = "\n".join([f"- {r['passage'][:500]} (score={r['score']:.2f})" for r in evidence_list[:3]])

        # Step 2: Use SLM to revise claim based on evidence
        prompt = f"""তুমি একজন তথ্য যাচাইকারী। নিচে একটি দাবি এবং কিছু প্রমাণ দেওয়া হলো। প্রমাণের ভিত্তিতে দাবিটি সংশোধন করো।

প্রশ্ন: {query}
মূল দাবি: {claim}

প্রমাণ:
{evidence_text}

নির্দেশনা:
- যদি প্রমাণ দাবিকে সমর্থন করে, দাবিটি অপরিবর্তিত রাখো
- যদি প্রমাণ দাবির বিরোধী হয়, প্রমাণ অনুযায়ী দাবি সংশোধন করো
- যদি প্রমাণ অপর্যাপ্ত হয়, বলো "প্রমাণ অপর্যাপ্ত"
- শুধু সংশোধিত দাবিটি বাংলায় দাও, ব্যাখ্যা নয়

সংশোধিত দাবি:"""

        try:
            corrected = self.slm.generate(prompt, max_new_tokens=256, temperature=0.2, do_sample=False)
            # Check if model says insufficient evidence
            if "প্রমাণ অপর্যাপ্ত" in corrected or "অপর্যাপ্ত" in corrected:
                status = "insufficient_evidence"
            elif corrected.strip() == claim.strip():
                status = "verified"
            else:
                status = "corrected"
        except Exception as e:
            print(f"[DataCorrector] Generation failed: {e}")
            corrected = claim
            status = "error"

        return {
            "original_claim": claim,
            "corrected_claim": corrected.strip(),
            "status": status,
            "evidence": evidence_list,
            "fallback_used": fallback_used,
            "retrieval_info": retrieval_out
        }

    def batch_correct(self, query: str, claims: List[str]) -> List[Dict]:
        return [self.correct_claim(query, c) for c in claims]
