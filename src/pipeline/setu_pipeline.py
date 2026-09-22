"""
SETU Full Pipeline - The main orchestration
Bengali query -> Draft -> Decompose -> Uncertainty -> Triage -> Correction -> Abstention -> Reassembly
"""
from typing import Dict, List
import time

class SETUPipeline:
    def __init__(self, slm_generator, claim_decomposer, uncertainty_scorer, triage_router,
                 data_corrector, reasoning_corrector, abstention_module, reassembly_module, config=None):
        self.slm = slm_generator
        self.decomposer = claim_decomposer
        self.uncertainty = uncertainty_scorer
        self.triage = triage_router
        self.data_corrector = data_corrector
        self.reasoning_corrector = reasoning_corrector
        self.abstention = abstention_module
        self.reassembly = reassembly_module
        self.config = config

    def run(self, query: str, context: str = None, task_type: str = "qa", language: str = "bn", verbose: bool = True) -> Dict:
        """
        Full SETU run for a single query
        """
        start_time = time.time()
        logs = []

        def log(msg):
            if verbose:
                print(msg)
            logs.append(msg)

        # Stage 1: Draft generation
        log(f"\n[SETU] Stage 1: Draft Generation | Query: {query[:100]}...")
        draft_prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer in Bengali:" if context else query
        draft_answer = self.slm.generate(draft_prompt, max_new_tokens=512, temperature=0.7)
        log(f"[Draft] {draft_answer}")

        # Stage 2: Atomic claim decomposition
        log(f"\n[SETU] Stage 2: Claim Decomposition")
        claims = self.decomposer.decompose(draft_answer, method="hybrid")
        log(f"[Claims] Found {len(claims)} claims: {claims}")

        if not claims:
            return {
                "query": query,
                "draft_answer": draft_answer,
                "final_answer": draft_answer,
                "claims": [],
                "logs": logs,
                "time": time.time() - start_time
            }

        # Stage 3 & 4: Uncertainty scoring + Triage + Correction per claim
        log(f"\n[SETU] Stage 3-5: Uncertainty -> Triage -> Correction")
        claim_results = []

        for idx, claim in enumerate(claims):
            log(f"\n--- Claim {idx+1}/{len(claims)}: {claim}")

            # Uncertainty
            uncertainty_res = self.uncertainty.score_claim(query, claim)
            log(f"[Uncertainty] {uncertainty_res}")

            # If low uncertainty, keep as is
            if not uncertainty_res["is_flagged"]:
                log(f"[Skip] Low uncertainty, keeping original")
                claim_results.append({
                    "original_claim": claim,
                    "corrected_claim": claim,
                    "final_claim": claim,
                    "status": "low_uncertainty_verified",
                    "uncertainty": uncertainty_res,
                    "abstained": False
                })
                continue

            # Triage
            triage_res = self.triage.triage(query, claim, method="hybrid")
            log(f"[Triage] {triage_res}")

            # Route to corrector
            if triage_res["label"] == "data-driven":
                log(f"[Routing] -> Data-driven Corrector (RAG)")
                correction_res = self.data_corrector.correct_claim(query, claim)
            else:
                log(f"[Routing] -> Reasoning-driven Corrector (Belief-Consistency)")
                correction_res = self.reasoning_corrector.correct_claim(query, claim)

            log(f"[Correction] {correction_res}")

            # Abstention check
            final_res = self.abstention.process(correction_res, language=language)
            final_res["uncertainty"] = uncertainty_res
            final_res["triage"] = triage_res
            log(f"[Final Claim] {final_res['final_claim']} | Abstained: {final_res['abstained']}")

            claim_results.append(final_res)

        # Stage 6: Reassembly
        log(f"\n[SETU] Stage 6: Reassembly")
        final_answer = self.reassembly.reassemble(query, draft_answer, claim_results, method="simple")
        log(f"[Final Answer] {final_answer}")

        total_time = time.time() - start_time

        return {
            "query": query,
            "context": context,
            "task_type": task_type,
            "language": language,
            "draft_answer": draft_answer,
            "claims": claims,
            "claim_results": claim_results,
            "final_answer": final_answer,
            "logs": logs,
            "time": total_time,
            "stats": {
                "n_claims": len(claims),
                "n_flagged": sum(1 for r in claim_results if r.get("uncertainty", {}).get("is_flagged", False)),
                "n_data_driven": sum(1 for r in claim_results if r.get("triage", {}).get("label") == "data-driven"),
                "n_reasoning_driven": sum(1 for r in claim_results if r.get("triage", {}).get("label") == "reasoning-driven"),
                "n_corrected": sum(1 for r in claim_results if r.get("status") == "corrected"),
                "n_abstained": sum(1 for r in claim_results if r.get("abstained", False)),
                "n_fallback": sum(1 for r in claim_results if r.get("fallback_used", False))
            }
        }

    def run_batch(self, instances: List[Dict], verbose: bool = False) -> List[Dict]:
        results = []
        for inst in instances:
            q = inst.get("query") or inst.get("question") or ""
            ctx = inst.get("context")
            task = inst.get("task_type", "qa")
            lang = inst.get("language", "bn")
            res = self.run(q, context=ctx, task_type=task, language=lang, verbose=verbose)
            results.append(res)
        return results
