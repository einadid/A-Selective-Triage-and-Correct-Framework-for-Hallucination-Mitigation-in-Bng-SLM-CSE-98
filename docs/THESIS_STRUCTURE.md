# Thesis Structure - SETU

## Chapter 1: Introduction
- 1.1 Background: LLMs hallucination, SLM more vulnerable
- 1.2 Motivation: Bengali 6th most spoken, low-resource, no mitigation
- 1.3 Problem Statement
- 1.4 Objectives (7 objectives from proposal)
- 1.5 Contributions (6 contributions)
- 1.6 Thesis Organization

## Chapter 2: Literature Review
- 2.1 Hallucination in LLMs (general)
- 2.2 Hallucination Evaluation: HaluEval, BenHalluEval (2026)
- 2.3 Uncertainty Signals: SelfCheckGPT, Semantic Entropy
- 2.4 Mitigation Strategies: RAG, Chain-of-Verification, HalluSearch
- 2.5 Decomposition: HalluGuard data vs reasoning taxonomy, BTPROP belief tree
- 2.6 Bengali NLP: BanglaBERT, BanglaCHQ, SOMADHAN, Cross-Lingual RAG
- 2.7 Research Gap Table (Table 1 from proposal) - SETU vs all

## Chapter 3: Proposed Methodology - SETU Framework
- 3.1 Overview Diagram (pipeline)
- 3.2 Draft Generation (SLM + 4-bit)
- 3.3 Atomic Claim Decomposition (Bengali-adapted)
- 3.4 Uncertainty Scoring (RQ1) - 3 signals + ECE/AUROC
- 3.5 Triage Router (RQ2) - rule + LLM + retrieval probe
- 3.6 Data-driven Correction (cross-lingual RAG)
- 3.7 Reasoning-driven Correction (BTPROP-lite + NLI)
- 3.8 Abstention Module (RQ4)
- 3.9 Reassembly
- 3.10 Datasets (TyDiQA, BanglaCHQ-Summ, SOMADHAN, BenHalluEval)

## Chapter 4: Experimental Setup & Results
- 4.1 Experimental Setup: Kaggle T4, 4-bit, seeds
- 4.2 Evaluation Metrics: BenHalluScore dual-track, Accuracy, Abstention rate, ECE
- 4.3 Baselines: Raw SLM, CoT, Uniform RAG, CoVe
- 4.4 RQ1 Results: Calibration Study (native vs code-mixed) - graphs
- 4.5 RQ2 Results: Triage Accuracy
- 4.6 RQ3 Results: SETU vs Baselines on 4 tasks (BenHalluScore table)
- 4.7 Ablation Study (5 ablations) - proves method novelty
- 4.8 RQ4 Results: Over-abstention analysis
- 4.9 Qualitative Examples (good and bad cases)
- 4.10 Error Analysis

## Chapter 5: Conclusion & Future Work
- 5.1 Summary
- 5.2 Contributions Revisited
- 5.3 Limitations (retrieval coverage, compute, NLI errors)
- 5.4 Future Work: fine-tune triage classifier, larger SLM, human eval, real deployment

## References
- Use BibTeX from verified papers (Group A, B, C from proposal)

## Appendix
- A. Prompt Templates (Bengali)
- B. Code Repository Link
- C. Sample Outputs
- D. FAISS Index Details

---

## Defense Presentation Structure (15 slides)
1. Title
2. Introduction & Motivation (Bengali + SLM problem)
3. Problem Statement
4. Literature Gap (Table 1)
5. Objectives
6. Proposed Methodology Diagram
7. Uncertainty Scoring (RQ1)
8. Triage Router (RQ2)
9. Correction Modules
10. Datasets & Evaluation (BenHalluScore)
11. Results: RQ1 Calibration
12. Results: RQ3 Main + Ablations
13. Results: Qualitative
14. Conclusion & Contributions
15. Thank You + Q&A

---

## Documentation to Produce
- [ ] README.md (done)
- [ ] requirements.txt (done)
- [ ] Thesis report (LaTeX/Word)
- [ ] Presentation PPT
- [ ] Kaggle notebooks (3 notebooks)
- [ ] Demo video (optional but good)
- [ ] Paper draft for BLP workshop
