# SETU Thesis - Complete Roadmap (বাংলায়)

## ১. ওরা কি করতে চাইছে? (What)

**SETU = Selective Evidence-grounded Triage with Uncertainty-aware correction**

একটা framework বানানো যা **বাংলা Small Language Model (1-3B) এর hallucination ঠিক করবে**, শুধু detect করবে না।

Pipeline:
```
বাংলা প্রশ্ন -> SLM Draft Answer -> Atomic Claim এ ভাগ -> Uncertainty দিয়ে সন্দেহজনক claim flag -> 
Triage: Data-driven vs Reasoning-driven আলাদা করা -> 
  - Data-driven হলে: Bengali Wikipedia + Cross-lingual English Retrieval দিয়ে ঠিক করা
  - Reasoning-driven হলে: Belief Consistency Check (NLI) দিয়ে ঠিক করা
-> যদি ঠিক না হয়: "আমি নিশ্চিত নই" বলে Abstain করা
-> Final Answer জোড়া লাগানো
```

## ২. কেন করছে? (Why)

1. **বাংলা ৬ষ্ঠ বৃহত্তম ভাষা** (৩০০M+ speaker) কিন্তু low-resource - কোনো hallucination mitigation framework নেই।
2. **SLM বেশি hallucinate করে** - বড় মডেলের চেয়ে ছোট মডেল (1-3B) অনেক বেশি ভুল করে, আর low-resource setting এ SLM ই deploy করতে হয়।
3. **Existing কাজ শুধু detection** - BenHalluEval (2026) শুধু মাপে, ঠিক করে না। BanglaForge শুধু code generation এ।
4. **Chain-of-Thought কাজ করে না** - BenHalluEval দেখিয়েছে CoT 21 টার মধ্যে 12 টা ক্ষেত্রে hallucination বাড়ায়!
5. **Code-mixed (বাংলা+English)** - বাস্তবে মানুষ "Dhaka er population koto?" এরকম লেখে, এটার জন্য কোনো mitigation নেই।

## ৩. করে লাভ কি? (Impact)

- **First-ever Bengali mitigation framework** - তোমার thesis হবে first
- **Publishable** - BLP / LoResLM workshop এ paper submit করা যাবে (proposal এ mention আছে)
- **Real-world safety** - Medical bot, summarizer, tutor এ ভুল তথ্য ঠেকাবে
- **Method-level novelty** - শুধু dataset swap না, নতুন mechanism:
  - Type-aware triage (data vs reasoning) - আগে কেউ করেনি
  - Uncertainty calibration study for Bengali - first
  - Cross-lingual fallback for sparse Bengali corpus
  - Abstention as first-class output
- **Deployable** - 4-bit quantization এ Kaggle T4/P100 তে চলবে

## ৪. কিভাবে Complete করবে? (How - 6 Months Plan)

### Month 1: Foundation
- [ ] Base papers পড়া: BenHalluEval, HalluGuard, BTPROP, HalluSearch, MFAVA, Cross-Lingual RAG
- [ ] Dataset download: TyDiQA-GoldP Bengali, BanglaCHQ-Summ, SOMADHAN, BenHalluEval 12K
- [ ] Wikipedia dump download + FAISS index build (BGE-M3 embeddings)
- [ ] Kaggle setup: Qwen2.5-3B 4-bit inference notebook চালু করা
- **Deliverable**: Data unified JSONL + FAISS index ready

### Month 2: RQ1 - Uncertainty Calibration Study (সবচেয়ে important, early risk mitigation)
- [ ] Claim decomposition implement (Bengali-adapted)
- [ ] 3 signals implement: Semantic Entropy, Self-Consistency, Verbalized Confidence
- [ ] NLI model (mDeBERTa-v3) দিয়ে semantic clustering
- [ ] Native vs Code-mixed এ calibration measure: ECE, AUROC
- [ ] Graph: কোন signal বাংলায় সবচেয়ে reliable?
- **Deliverable**: RQ1 result - standalone publishable finding

### Month 3: RQ2 - Triage Router + Data-driven Fix
- [ ] Triage classifier: rule-based + LLM-based + retrieval probe
- [ ] Data-driven correction: Bengali FAISS + cross-lingual fallback (bn->en translate -> enwiki retrieve)
- [ ] Evaluation: Triage accuracy কত?
- **Deliverable**: Triage module working

### Month 4: Reasoning Fix + Abstention + Full Integration
- [ ] Reasoning-driven correction: BTPROP-lite (sub-claim generate + NLI consistency)
- [ ] Abstention module: "আমি নিশ্চিত নই"
- [ ] Reassembly module
- [ ] Full SETU pipeline end-to-end first run -> BenHalluScore
- **Deliverable**: End-to-end pipeline

### Month 5: Baselines, Ablations, Stats (Thesis এর প্রমাণ)
- [ ] 4 baselines: Raw SLM, CoT, Uniform RAG, Chain-of-Verification
- [ ] 5 ablations: No decomposition, No triage, No cross-lingual, No belief-consistency, No abstention
- [ ] 4 tasks এ evaluation: QA, Code-Mixed QA, Summarization, Reasoning
- [ ] Statistical significance test (paired t-test, multiple seeds)
- [ ] Error analysis + over-abstention study (RQ4)
- **Deliverable**: Results table + graphs

### Month 6: Thesis Writing + Defense
- [ ] Chapter 1-5 লেখা: Introduction, Literature, Methodology, Results, Conclusion
- [ ] Presentation slides
- [ ] Optional: BLP workshop paper submit
- **Deliverable**: Final thesis

## ৫. Technical Stack (Kaggle Friendly)

- **SLM**: Qwen2.5-1.5B/3B-Instruct, Gemma-2-2B-it, Llama-3.2-1B (4-bit via bitsandbytes)
- **Embeddings**: BGE-M3 / multilingual-e5-large
- **Vector DB**: FAISS CPU
- **NLI**: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual)
- **Retrieval**: BM25 + Dense hybrid
- **Datasets**: HuggingFace `datasets` library
- **Eval**: BenHalluScore dual-track

## ৬. তোমার এখন কি করতে হবে?

1. এই repo তে আমি full code skeleton বানিয়ে দিয়েছি - `src/` folder দেখো
2. Kaggle এ notebook খুলে `pip install -r requirements.txt`
3. `python src/data/loaders.py` দিয়ে data download শুরু করো
4. `python src/main.py --input "তোমার প্রশ্ন"` দিয়ে pipeline test করো
5. আমাকে বলো কোন step এ আটকাচ্ছো, আমি fix করে দেব

## ৭. Thesis Defense এর One-Sentence Claim

"To the best of our knowledge, SETU is the first framework to reduce (not merely detect) hallucination in Bengali small language models, using a novel uncertainty-triggered triage that routes each flagged claim to type-specific correction — cross-lingual retrieval for knowledge gaps, belief-consistency for reasoning errors — and abstains when unresolved, evaluated with dual-track BenHalluScore across native and code-mixed Bengali."

এই লাইনটা defense এ মুখস্থ রাখবে।

---

Need help? Ask me about any module!
