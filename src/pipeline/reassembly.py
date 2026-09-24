"""
Reassembly - Combine verified/corrected/abstained claims into final answer
"""
from typing import List, Dict

class ReassemblyModule:
    def __init__(self, slm_generator=None, config=None):
        self.slm = slm_generator
        self.config = config

    def simple_reassembly(self, original_answer: str, claim_results: List[Dict]) -> str:
        """Simple concatenation of final claims"""
        final_claims = [r.get("final_claim", r.get("corrected_claim", r.get("original_claim",""))) for r in claim_results]
        # Filter empty
        final_claims = [c for c in final_claims if c.strip()]

        # Join with proper punctuation
        reassembled = " ".join(final_claims)
        # Clean up double dandas
        reassembled = reassembled.replace("।।", "।").replace("  ", " ")
        return reassembled.strip()

    def llm_reassembly(self, query: str, original_answer: str, claim_results: List[Dict]) -> str:
        """Use SLM to fluently reassemble"""
        if self.slm is None:
            return self.simple_reassembly(original_answer, claim_results)

        final_claims_text = "\n".join([f"{i+1}. {r.get('final_claim','')}" for i, r in enumerate(claim_results)])

        prompt = f"""প্রশ্ন: {query}
মূল উত্তর: {original_answer}

যাচাইকৃত/সংশোধিত দাবিগুলো:
{final_claims_text}

এই যাচাইকৃত দাবিগুলো ব্যবহার করে একটি সুসংগত, সাবলীল বাংলা উত্তর তৈরি করো। 
যদি কোনো দাবিতে "আমি নিশ্চিত নই" থাকে, সেটি উত্তরে অন্তর্ভুক্ত করো।
উত্তরটি প্রাকৃতিক এবং পূর্ণ বাক্যে হওয়া উচিত।

চূড়ান্ত উত্তর:"""

        try:
            final = self.slm.generate(prompt, max_new_tokens=512, temperature=0.3, do_sample=False)
            return final.strip()
        except Exception as e:
            print(f"[Reassembly] LLM failed: {e}")
            return self.simple_reassembly(original_answer, claim_results)

    def reassemble(self, query: str, original_answer: str, claim_results: List[Dict], method: str = "simple") -> str:
        if method == "llm":
            return self.llm_reassembly(query, original_answer, claim_results)
        else:
            return self.simple_reassembly(original_answer, claim_results)
