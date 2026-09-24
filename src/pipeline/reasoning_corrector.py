"""
Reasoning-driven Corrector - Lightweight BTPROP (Belief Tree Propagation)
Expand claim into logically related statements and check consistency via NLI
"""
from typing import Dict, List

class ReasoningCorrector:
    def __init__(self, slm_generator, nli_model, config=None):
        self.slm = slm_generator
        self.nli = nli_model
        self.config = config

    def generate_subclaims(self, query: str, claim: str, n: int = 3) -> List[str]:
        """Generate logically related sub-claims (belief tree shallow)"""
        prompt = f"""প্রশ্ন: {query}
মূল দাবি: {claim}

এই দাবিটি যাচাই করার জন্য ৩টি যৌক্তিক উপ-দাবি বা প্রমাণ তৈরি করো যা এই দাবিকে সমর্থন করবে বা এর সাথে সম্পর্কিত।
প্রতিটি উপ-দাবি একটি পৃথক লজিক্যাল স্টেপ হবে।

উদাহরণ:
মূল দাবি: ৫+৩=৮
উপ-দাবি:
1. ৫ একটি সংখ্যা
2. ৩ একটি সংখ্যা
3. ৫+৩ যোগ করলে ৮ হয়

এখন তোমার উপ-দাবিগুলো দাও (শুধু ৩টি, নম্বর সহ):

উপ-দাবি:"""

        try:
            out = self.slm.generate(prompt, max_new_tokens=256, temperature=0.5, do_sample=True)
            subclaims = []
            for line in out.split("\n"):
                line = line.strip()
                # Remove numbering
                import re
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if len(line) > 10:
                    subclaims.append(line)
            return subclaims[:n]
        except Exception as e:
            print(f"[ReasoningCorrector] Subclaim gen failed: {e}")
            return []

    def check_consistency(self, claim: str, subclaims: List[str]) -> Dict:
        """Check if claim is consistent with its subclaims via NLI"""
        if not subclaims:
            return {"consistent": False, "scores": [], "reason": "no_subclaims"}

        scores = []
        contradictions = 0
        entailments = 0

        for sub in subclaims:
            try:
                # Does subclaim entail main claim? Or vice versa?
                # For reasoning, we check if subclaims together support claim
                # Use NLI both ways
                label1, conf1 = self.nli.predict(sub, claim)  # sub -> claim
                label2, conf2 = self.nli.predict(claim, sub)  # claim -> sub

                # If sub contradicts claim, inconsistency
                if label1 == "contradiction" and conf1 > 0.6:
                    contradictions += 1
                if label1 == "entailment" and conf1 > 0.6:
                    entailments += 1

                scores.append({
                    "subclaim": sub,
                    "sub->claim": (label1, conf1),
                    "claim->sub": (label2, conf2)
                })
            except Exception as e:
                print(f"[Consistency] NLI failed: {e}")
                continue

        # Decision: if contradictions > entailments -> inconsistent
        consistent = contradictions == 0 and entailments > 0
        # More lenient: if at least 1 entailment and no contradiction
        if entailments >= 1 and contradictions == 0:
            consistent = True
        elif contradictions >= 1:
            consistent = False
        else:
            consistent = False  # neutral - not enough support

        return {
            "consistent": consistent,
            "contradictions": contradictions,
            "entailments": entailments,
            "scores": scores
        }

    def correct_claim(self, query: str, claim: str) -> Dict:
        """
        Correct reasoning-driven hallucination via belief-consistency
        """
        # Step 1: Generate subclaims
        subclaims = self.generate_subclaims(query, claim, n=3)

        if not subclaims:
            return {
                "original_claim": claim,
                "corrected_claim": claim,
                "status": "no_subclaims",
                "subclaims": [],
                "consistency": {}
            }

        # Step 2: Check consistency
        consistency_res = self.check_consistency(claim, subclaims)

        if consistency_res["consistent"]:
            return {
                "original_claim": claim,
                "corrected_claim": claim,
                "status": "verified",
                "subclaims": subclaims,
                "consistency": consistency_res
            }

        # Step 3: If inconsistent, ask SLM to revise based on subclaims
        subclaim_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(subclaims)])
        prompt = f"""প্রশ্ন: {query}
মূল দাবি: {claim}

এই দাবিটি যৌক্তিকভাবে অসঙ্গতিপূর্ণ বলে মনে হচ্ছে। নিচে এর সাথে সম্পর্কিত উপ-দাবিগুলো দেওয়া হলো:

{subclaim_text}

উপরের উপ-দাবিগুলোর ভিত্তিতে মূল দাবিটি সংশোধন করো যাতে এটি যৌক্তিকভাবে সঙ্গতিপূর্ণ হয়।
শুধু সংশোধিত দাবিটি বাংলায় দাও:

সংশোধিত দাবি:"""

        try:
            corrected = self.slm.generate(prompt, max_new_tokens=256, temperature=0.3, do_sample=False)
            status = "corrected" if corrected.strip() != claim.strip() else "unresolved"
        except Exception as e:
            print(f"[ReasoningCorrector] Correction failed: {e}")
            corrected = claim
            status = "error"

        return {
            "original_claim": claim,
            "corrected_claim": corrected.strip(),
            "status": status,
            "subclaims": subclaims,
            "consistency": consistency_res
        }

    def batch_correct(self, query: str, claims: List[str]) -> List[Dict]:
        return [self.correct_claim(query, c) for c in claims]
