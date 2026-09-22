"""
Triage Router - RQ2: Classify hallucination as data-driven vs reasoning-driven
Based on HalluGuard taxonomy but black-box & lightweight
"""
from typing import Dict, Literal
import re

class TriageRouter:
    def __init__(self, config=None, slm_generator=None, retriever=None):
        self.config = config
        self.slm = slm_generator
        self.retriever = retriever

        self.data_keywords = config.triage.data_keywords if config else [
            "কবে", "কোথায়", "কে", "কত", "সাল", "তারিখ", "রাজধানী", "জনসংখ্যা",
            "capital", "date", "year", "population"
        ]
        self.reasoning_keywords = config.triage.reasoning_keywords if config else [
            "কেন", "কিভাবে", "কারণ", "যদি", "তাহলে", "অতএব", "সুতরাং",
            "why", "how", "therefore", "calculate"
        ]

    def rule_based_triage(self, claim: str) -> Dict:
        """Fast rule-based classifier"""
        claim_lower = claim.lower()

        data_score = 0
        reasoning_score = 0

        # Keyword matching
        for kw in self.data_keywords:
            if kw.lower() in claim_lower:
                data_score += 1

        for kw in self.reasoning_keywords:
            if kw.lower() in claim_lower:
                reasoning_score += 1

        # Patterns
        # Data-driven patterns: numbers, dates, entities
        if re.search(r'\d{4}\s*সাল|\d+\s*(জন|টাকা|কিলোমিটার|বছর)|১৯\d{2}|২০\d{2}', claim):
            data_score += 2

        if re.search(r'রাজধানী|অবস্থিত|জন্ম|মৃত্যু|আবিষ্কার|capital|located|born', claim_lower):
            data_score += 1

        # Reasoning-driven patterns: math, logic connectors
        if re.search(r'[+\-*/=]|যোগ|বিয়োগ|গুণ|ভাগ|যদি.*তাহলে|কারণ.*তাই', claim_lower):
            reasoning_score += 2

        if re.search(r'অতএব|সুতরাং|therefore|hence|thus|কারণে', claim_lower):
            reasoning_score += 1

        # Decision
        if data_score > reasoning_score:
            label = "data-driven"
            conf = min(0.5 + (data_score - reasoning_score)*0.15, 0.95)
        elif reasoning_score > data_score:
            label = "reasoning-driven"
            conf = min(0.5 + (reasoning_score - data_score)*0.15, 0.95)
        else:
            # Tie -> check if claim contains inferential language
            label = "data-driven"  # default to retrieval, safer
            conf = 0.5

        return {
            "label": label,
            "confidence": conf,
            "data_score": data_score,
            "reasoning_score": reasoning_score,
            "method": "rule"
        }

    def llm_based_triage(self, query: str, claim: str) -> Dict:
        """Use SLM itself to classify"""
        if self.slm is None:
            return self.rule_based_triage(claim)

        prompt = f"""তুমি একজন hallucination triage বিশেষজ্ঞ। একটি দাবি দেওয়া হবে, তোমাকে বলতে হবে এটি কোন ধরনের ভুল:

1. data-driven: বাইরের তথ্য/জ্ঞানের অভাব - তারিখ, স্থান, ব্যক্তি, সংখ্যা, তথ্যগত ভুল
2. reasoning-driven: যুক্তি/গণনার ভুল - গাণিতিক ভুল, লজিক্যাল ভুল, ভুল সিদ্ধান্ত

প্রশ্ন: {query}
দাবি: {claim}

শুধু "data-driven" অথবা "reasoning-driven" উত্তর দাও, সাথে 0-1 confidence, format: <label> | <confidence>
যেমন: data-driven | 0.85
উত্তর:"""

        try:
            out = self.slm.generate(prompt, max_new_tokens=20, temperature=0.1, do_sample=False)
            # Parse
            out_lower = out.lower()
            label = "data-driven" if "data" in out_lower else "reasoning-driven" if "reasoning" in out_lower else "data-driven"
            # Extract confidence
            import re
            m = re.search(r'0?\.\d+|1\.0', out)
            conf = float(m.group(0)) if m else 0.6
            return {
                "label": label,
                "confidence": conf,
                "method": "llm",
                "raw_output": out
            }
        except Exception as e:
            print(f"[Triage] LLM failed: {e}")
            return self.rule_based_triage(claim)

    def retrieval_probe_triage(self, claim: str) -> Dict:
        """Grounded in HalluGuard idea: if claim is retrieval-answerable -> data-driven"""
        if self.retriever is None:
            return self.rule_based_triage(claim)

        try:
            # Try to retrieve
            results = self.retriever.retrieve(claim, top_k=3)
            # If high retrieval score, likely data-driven (needs external knowledge)
            # If low retrieval score but claim is about reasoning, likely reasoning-driven
            max_score = max([r.get("score", 0) for r in results]) if results else 0

            if max_score > 0.6:
                return {"label": "data-driven", "confidence": max_score, "method": "retrieval_probe", "retrieval_score": max_score}
            else:
                # Check if claim has reasoning structure
                rule = self.rule_based_triage(claim)
                if rule["reasoning_score"] > 0:
                    return {"label": "reasoning-driven", "confidence": 0.6, "method": "retrieval_probe", "retrieval_score": max_score}
                else:
                    return {"label": "data-driven", "confidence": 0.55, "method": "retrieval_probe", "retrieval_score": max_score}
        except Exception as e:
            print(f"[Triage] Retrieval probe failed: {e}")
            return self.rule_based_triage(claim)

    def triage(self, query: str, claim: str, method: str = "hybrid") -> Dict:
        """
        Main triage function
        method: rule | llm | retrieval | hybrid
        """
        if method == "rule":
            return self.rule_based_triage(claim)
        elif method == "llm":
            return self.llm_based_triage(query, claim)
        elif method == "retrieval":
            return self.retrieval_probe_triage(claim)
        else:  # hybrid: combine signals
            rule_res = self.rule_based_triage(claim)
            # If retriever available, use it
            if self.retriever:
                retr_res = self.retrieval_probe_triage(claim)
                # Weighted vote: if both agree, high conf; if disagree, use higher conf
                if rule_res["label"] == retr_res["label"]:
                    return {
                        "label": rule_res["label"],
                        "confidence": (rule_res["confidence"] + retr_res["confidence"])/2,
                        "method": "hybrid",
                        "components": [rule_res, retr_res]
                    }
                else:
                    # Prefer retrieval probe for data-driven detection
                    if retr_res["confidence"] > rule_res["confidence"]:
                        return {**retr_res, "method": "hybrid", "components": [rule_res, retr_res]}
                    else:
                        return {**rule_res, "method": "hybrid", "components": [rule_res, retr_res]}
            else:
                # No retriever, try LLM if available
                if self.slm:
                    llm_res = self.llm_based_triage(query, claim)
                    if rule_res["label"] == llm_res["label"]:
                        return {
                            "label": rule_res["label"],
                            "confidence": (rule_res["confidence"] + llm_res["confidence"])/2,
                            "method": "hybrid",
                            "components": [rule_res, llm_res]
                        }
                    else:
                        return {**llm_res, "method": "hybrid", "components": [rule_res, llm_res]}
                else:
                    return rule_res

if __name__ == "__main__":
    router = TriageRouter()
    print(router.triage("বাংলাদেশের রাজধানী কোথায়?", "ঢাকা বাংলাদেশের রাজধানী।"))
    print(router.triage("৫+৩=?", "৫+৩=৯ কারণ ২+২=৪ তাই ৫+৩=৯।"))
