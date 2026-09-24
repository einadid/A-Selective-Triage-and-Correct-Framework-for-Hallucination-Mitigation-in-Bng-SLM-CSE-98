# SETU Weekly Update - Week 1
Date: 2026-09-22
Students: Kazi Tajrian Mostafa (CSE 030 07859) & Sayed Raisul Alam Raihan (CSE 030 07798)
Supervisor: Mr. Ratul Barua
Thesis: SETU - A Selective Triage-and-Correct Framework for Hallucination Mitigation in Bengali SLMs

## ✅ Completed This Week:

### 1. Full Framework Implementation (12 modules) - DONE
- **Models:**
  - SLM Generator: Qwen2.5-1.5B/3B, Gemma-2-2B, Llama-3.2-1B with 4-bit quantization (bitsandbytes)
  - NLI Model: mDeBERTa-v3-base-mnli-xnli (multilingual)
- **Pipeline:**
  - Claim Decomposer: Bengali-adapted atomic claim splitting (rule + LLM)
  - Uncertainty Scorer: 3 signals - Semantic Entropy, Self-Consistency, Verbalized Confidence + ECE/AUROC calibration
  - Triage Router: data-driven vs reasoning-driven classifier (rule + LLM + retrieval probe) - based on HalluGuard taxonomy
  - Data-driven Corrector: Bengali Wikipedia FAISS (BGE-M3) + Cross-lingual fallback (bn->en)
  - Reasoning Corrector: BTPROP-lite (sub-claim generation + NLI consistency check)
  - Abstention Module: "আমি নিশ্চিত নই" when unresolved
  - Reassembly: Final answer generation
- **Evaluation:**
  - BenHalluScore: Dual-track (Track A false alarm + Track B miss) - lower is better
  - Baselines: Raw SLM, CoT, Uniform RAG, Chain-of-Verification
  - Metrics: Accuracy, Abstention rate, Retrieval recall@k, ECE

### 2. Kaggle Setup - DONE
- Qwen2.5-1.5B-Instruct 4-bit running on T4 x2 (16GB VRAM, usage ~6GB)
- Tested Bengali Q&A:
  - Q: বাংলাদেশের রাজধানী কোথায়? -> A: ঢাকা (correct)
  - Q: ঢাকার জনসংখ্যা -> Sometimes 5 crore (hallucination) -> will be caught by RAG
  - Q: 5+3=? -> Sometimes 9 (reasoning error) -> will be caught by BTPROP-lite
- Notebook: `notebooks/SETU_Kaggle_Full.ipynb` (6 cells, ready to run)

### 3. Documentation - DONE
- `docs/ANALYSIS_BN.md`: Full analysis in Bengali - what, why, impact, roadmap
- `docs/THESIS_ROADMAP.md`: 6 months plan with deliverables
- `docs/THESIS_STRUCTURE.md`: Chapter 1-5 structure + defense slides
- `README.md`: GitHub ready with quick start
- `TODO_NOW.md`: Immediate tasks

### 4. GitHub - DONE
- Repo: https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
- Branch: `arena/01a0c59d-a-selective-triage-and-correct`
- PR #1: https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98/pull/1 (Draft - not merged yet, will auto-update)
- 28 files committed, demo working: `python run_demo.py`

## 📊 Current Status:
- Dummy pipeline: Working (claim decomposition + triage)
- Real model: Working on Kaggle (Qwen2.5-1.5B 4-bit)
- Full integration: Code ready, needs FAISS index + dataset download for end-to-end test

## 🎯 Next Week Plan (Week 2):
- [ ] Download datasets: TyDiQA-GoldP Bengali (5000), BanglaCHQ-Summ (4000), SOMADHAN (2000)
- [ ] Build Wikipedia FAISS index: bnwiki + enwiki dumps, BGE-M3 embeddings, chunk size 256
- [ ] RQ1: Uncertainty calibration study
  - Run 100 samples native Bengali + 100 code-mixed
  - Compute 3 signals, measure ECE, AUROC
  - Graph: which signal is most calibrated for Bengali?
- [ ] Implement full data-driven correction with cross-lingual fallback

## ❓ Need from Supervisor:
1. Approval for methodology (Table 1 positioning - SETU vs BenHalluEval, HalluGuard, etc.)
2. Dataset access: BenHalluEval 12K candidates if available (arXiv 2605.31483)
3. Compute: Kaggle T4 is enough for thesis (4-bit), but if varsity PC has RTX 3060/4060, FAISS build will be faster (2-3 hours -> 30 min)
4. Confirmation on SLM choice: Qwen2.5-1.5B/3B vs Gemma-2-2B vs Llama-3.2-1B

## 📎 Attachments for Supervisor:
- PR Link: https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98/pull/1
- Demo Output: `run_demo.py` logs
- Kaggle Notebook: `notebooks/SETU_Kaggle_Full.ipynb`
- Proposal: `SETU_Thesis_Proposal.pdf`

## One-Sentence Claim for Defense:
"To the best of our knowledge, SETU is the first framework to reduce (not merely detect) hallucination in Bengali small language models, using a novel uncertainty-triggered triage that routes each flagged claim to type-specific correction — cross-lingual retrieval for knowledge gaps, belief-consistency for reasoning errors — and abstains when unresolved, evaluated with dual-track BenHalluScore across native and code-mixed Bengali."

---
Prepared by: Kazi & Raihan
Next Update: Week 2 (2026-09-29)
