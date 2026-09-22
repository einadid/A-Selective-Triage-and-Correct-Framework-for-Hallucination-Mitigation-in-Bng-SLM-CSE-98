"""
Atomic Claim Decomposition - Bengali adapted from HalluSearch / FActScore
Splits answer into single-fact claims
"""
import re
from typing import List

class BengaliClaimDecomposer:
    def __init__(self, slm_generator=None):
        self.slm = slm_generator
        # Bengali sentence delimiters
        self.delimiters = r'[।!?।\n]+'

    def rule_based_split(self, text: str) -> List[str]:
        """Simple rule-based splitter for Bengali"""
        # Clean
        text = text.strip()
        if not text:
            return []

        # Split by Bengali danda and punctuation
        sentences = re.split(self.delimiters, text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            # Further split by conjunctions that often separate facts
            # e.g., "এবং", "আর", "যা"
            sub_parts = re.split(r'\s+এবং\s+|\s+আর\s+|\s+যা\s+|\s+,\s+', sent)
            for part in sub_parts:
                part = part.strip()
                if len(part) > 10:  # filter very short
                    # Add back punctuation for completeness
                    if not part.endswith('।'):
                        part = part + '।'
                    claims.append(part)

        # Deduplicate
        seen = set()
        uniq = []
        for c in claims:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def llm_based_split(self, text: str) -> List[str]:
        """Use SLM to decompose into atomic claims - more accurate"""
        if self.slm is None:
            return self.rule_based_split(text)

        prompt = f"""তোমাকে একটি বাংলা উত্তর দেওয়া হবে। এটিকে একক তথ্য-ভিত্তিক দাবিতে (atomic claims) ভাগ করো।
প্রতিটি দাবি একটি মাত্র তথ্য ধারণ করবে।

উদাহরণ:
ইনপুট: "ঢাকা বাংলাদেশের রাজধানী এবং এটি বুড়িগঙ্গা নদীর তীরে অবস্থিত। এর জনসংখ্যা প্রায় ২ কোটি।"
আউটপুট:
1. ঢাকা বাংলাদেশের রাজধানী।
2. ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।
3. ঢাকার জনসংখ্যা প্রায় ২ কোটি।

এখন এই উত্তরটি ভাগ করো:
"{text}"

শুধু নম্বর সহ দাবিগুলো দাও, অন্য কিছু নয়:"""

        try:
            out = self.slm.generate(prompt, max_new_tokens=512, temperature=0.2, do_sample=False)
            # Parse numbered list
            claims = []
            for line in out.split("\n"):
                line = line.strip()
                # Remove numbering like "1. "
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = line.strip()
                if len(line) > 10:
                    claims.append(line)
            if len(claims) >= 1:
                return claims
        except Exception as e:
            print(f"[Decomposer] LLM split failed: {e}")

        return self.rule_based_split(text)

    def decompose(self, text: str, method: str = "hybrid") -> List[str]:
        """
        method: rule | llm | hybrid (try llm, fallback to rule)
        """
        if method == "rule":
            return self.rule_based_split(text)
        elif method == "llm":
            return self.llm_based_split(text)
        else:  # hybrid
            claims = self.llm_based_split(text)
            # If LLM returns too few claims relative to sentences, fallback to rule
            rule_claims = self.rule_based_split(text)
            if len(claims) < len(rule_claims) * 0.5:
                return rule_claims
            return claims

if __name__ == "__main__":
    decomposer = BengaliClaimDecomposer()
    text = "ঢাকা বাংলাদেশের রাজধানী এবং এটি বুড়িগঙ্গা নদীর তীরে অবস্থিত। এর জনসংখ্যা প্রায় ২ কোটি। বাংলাদেশ ১৯৭১ সালে স্বাধীন হয়।"
    print(decomposer.decompose(text, method="rule"))
