# SETU: Selective Triage-and-Correct Framework for Hallucination Mitigation in Bengali SLMs

SETU (সেতু = bridge) is the first framework to **mitigate** (not just detect) hallucinations in Bengali Small Language Models.

Thesis: CSE, Port City International University
Authors: Kazi Tajrian Mostafa & Sayed Raisul Alam Raihan
Supervisor: Mr. Ratul Barua

## Core Idea
Bengali SLM -> Draft Answer -> Atomic Claim Decomposition -> Uncertainty Scoring -> Triage (Data-driven vs Reasoning-driven) -> Targeted Correction -> Abstention -> Final Answer

## Pipeline
1.  **Draft Generation**: Qwen2.5-3B, Gemma-2-2B, Llama3.2-1B at 4-bit
2.  **Atomic Claim Decomposition**: Bengali-adapted claim splitter
3.  **Uncertainty Scoring**: Semantic Entropy, Self-Consistency, Verbalized Confidence + ECE/AUROC calibration
4.  **Triage Router**: Lightweight classifier -> data-driven (needs external facts) vs reasoning-driven (logic error)
5.  **Data-driven Fix**: Bengali Wikipedia FAISS (BGE-M3) + Cross-lingual fallback (Bengali -> English retrieval)
6.  **Reasoning-driven Fix**: Shallow Belief-Consistency Check (BTPROP-lite + NLI)
7.  **Abstention**: "আমি নিশ্চিত নই" when unresolved
8.  **Reassembly**: Final grounded Bengali answer

## Evaluation
- Primary: Dual-Track BenHalluScore = 0.5*(Track A error + Track B error) - lower is better
- Tasks: TyDiQA-GoldP (QA), Code-Mixed QA, BanglaCHQ-Summ (Summarization), SOMADHAN (Reasoning)
- Baselines: Raw SLM, CoT, Uniform RAG, Chain-of-Verification
- Ablations: No decomposition, No triage, No cross-lingual, No belief-consistency, No abstention

## Quick Start
```bash
pip install -r requirements.txt
python src/main.py --task qa --model qwen2.5-3b --input "বাংলাদেশের রাজধানী কোথায়?"
```

See `docs/` for full thesis documentation.
