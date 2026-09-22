"""
Abstention Module - RQ4: When to say "আমি নিশ্চিত নই"
"""
from typing import Dict

class AbstentionModule:
    def __init__(self, config=None):
        self.phrase = config.evaluation.abstention_phrase if config else "আমি নিশ্চিত নই"
        # Could also have English fallback for code-mixed
        self.phrase_en = "I am not sure"

    def should_abstain(self, correction_result: Dict) -> bool:
        """
        Decide if we should abstain based on correction result
        """
        status = correction_result.get("status", "")

        # Abstain if:
        # - No evidence found
        # - Insufficient evidence
        # - Unresolved after correction attempts
        # - Error during correction
        abstain_statuses = ["no_evidence", "insufficient_evidence", "unresolved", "error", "no_subclaims"]

        if status in abstain_statuses:
            return True

        # Also check if corrected claim still has low confidence
        # (could integrate uncertainty scorer here)
        corrected = correction_result.get("corrected_claim", "")
        if not corrected or len(corrected.strip()) < 5:
            return True

        # If fallback was used and still low score
        retrieval_info = correction_result.get("retrieval_info", {})
        if retrieval_info.get("fallback_used") and retrieval_info.get("max_score", 0) < 0.2:
            return True

        return False

    def abstain_claim(self, original_claim: str, language: str = "bn") -> str:
        """Return abstention phrase"""
        if language == "bn-en-code-mixed":
            return f"{self.phrase} / {self.phrase_en} regarding '{original_claim[:50]}...'"
        return self.phrase

    def process(self, correction_result: Dict, language: str = "bn") -> Dict:
        """Process correction result and apply abstention if needed"""
        if self.should_abstain(correction_result):
            return {
                **correction_result,
                "final_claim": self.abstain_claim(correction_result.get("original_claim",""), language),
                "abstained": True
            }
        else:
            return {
                **correction_result,
                "final_claim": correction_result.get("corrected_claim", correction_result.get("original_claim","")),
                "abstained": False
            }
