# SETU Thesis - Full Analysis (Boss er jonno)

## তোমার Thesis টা আসলে কি?

**Title:** SETU: A Selective Triage-and-Correct Framework for Hallucination Mitigation in Bengali Small Language Models

**সহজ বাংলায়:**
তুমি একটা AI system বানাবে যেটা ছোট বাংলা AI মডেল (1-3 Billion parameter) এর ভুলভাল উত্তর ধরে ঠিক করে দেবে।

যেমন:
- User: "ঢাকার জনসংখ্যা কত?"
- ছোট মডেল ভুল বলল: "৫ কোটি" (আসলে ২.২ কোটি)
- তোমার SETU framework সেটা ধরে বলবে: "না, ২.২ কোটি" - Wikipedia দেখে প্রমাণ সহ

এটা **প্রথমবার** বাংলায় কেউ করছে। আগে সবাই শুধু ভুল ধরতো (detection), কেউ ঠিক করতো না (correction).

---

## ওরা কি করতে চাইছে? (Detailed Breakdown)

### 1. Draft Generation (SLM দিয়ে প্রথম উত্তর)
- Model: Qwen2.5-3B, Gemma-2-2B, Llama-3.2-1B
- 4-bit quantization এ Kaggle T4 GPU তে চালাবে
- প্রথমে একটা draft answer বানাবে - যেটাতে hallucination থাকতে পারে

### 2. Atomic Claim Decomposition
- Draft answer কে ছোট ছোট একক তথ্যে ভাগ করা
- যেমন: "ঢাকা বাংলাদেশের রাজধানী এবং এটি বুড়িগঙ্গা নদীর তীরে অবস্থিত।"
  -> Claim 1: ঢাকা বাংলাদেশের রাজধানী।
  -> Claim 2: ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।
- কেন? প্রতিটা claim আলাদা ভাবে verify করা যায়

### 3. Uncertainty Scoring (RQ1 - তোমার First Contribution)
- ৩টা signal দিয়ে মাপবে কোন claim টা সন্দেহজনক:
  1. **Semantic Entropy**: একই প্রশ্ন ৭ বার করে দেখবে মডেল একই মানে উত্তর দেয় কিনা। যদি ৭ বার ৭ রকম বলে, entropy high = hallucination
  2. **Self-Consistency**: NLI model দিয়ে check করবে sample গুলো একে অপরকে support করে কিনা
  3. **Verbalized Confidence**: মডেলকে জিজ্ঞেস করবে "তোমার confidence কত?" - 0.85 বললে confident, 0.3 বললে uncertain
- **Calibration Study**: বাংলায় কোন signal টা সবচেয়ে reliable? Native বাংলা vs Code-mixed (Bangla+English) এ পার্থক্য আছে কিনা? এটা first study!

### 4. Triage Router (RQ2 - Main Novelty)
- Hallucination কে ২ ভাগে ভাগ করা (HalluGuard paper থেকে idea):
  - **Data-driven**: তথ্যের অভাব - তারিখ, স্থান, ব্যক্তি, সংখ্যা ভুল
    - উদা: "বাংলাদেশ ১৯৮০ সালে স্বাধীন হয়" (ভুল তারিখ)
  - **Reasoning-driven**: যুক্তির ভুল - গণিত, লজিক ভুল
    - উদা: "৫+৩=৯ কারণ ২+২=৪" (logic ভুল)
- Lightweight classifier দিয়ে আলাদা করবে - black-box, model এর ভিতরে ঢুকতে হবে না

### 5. Targeted Correction (RQ3)
- **Data-driven হলে**: Cross-lingual RAG
  - প্রথমে Bengali Wikipedia FAISS এ search করবে (BGE-M3 embedding)
  - যদি Bengali তে না পায় (sparse corpus), English এ translate করে English Wikipedia search করবে, তারপর বাংলায় ফিরিয়ে আনবে
  - Evidence দিয়ে claim ঠিক করবে
  
- **Reasoning-driven হলে**: Belief-Consistency Check (BTPROP-lite)
  - Claim থেকে ৩টা sub-claim বানাবে
  - NLI দিয়ে check করবে sub-claim গুলো main claim কে support করে কিনা
  - যদি contradiction থাকে, SLM কে দিয়ে revise করাবে

### 6. Abstention (RQ4)
- যদি কোনোভাবেই ঠিক করা না যায়, মডেল বলবে: **"আমি নিশ্চিত নই"**
- ভুল তথ্য দেওয়ার চেয়ে "জানি না" বলা ভালো - 2026 এর trend
- Over-abstention measure করবে - বেশি "জানি না" বলছে কিনা

### 7. Reassembly
- সব verified/corrected/abstained claim জোড়া লাগিয়ে final answer বানাবে

### 8. Evaluation
- **Primary**: BenHalluScore (BenHalluEval থেকে)
  - Track A: সঠিক উত্তরে false alarm করে কিনা (ভুলে "hallucination" বলে কিনা)
  - Track B: ভুল উত্তরে miss করে কিনা (hallucination ধরতে পারে কিনা)
  - Score = 0.5*(Track A error + Track B error) - lower better
- **Tasks**: 
  - QA: TyDiQA-GoldP Bengali
  - Code-Mixed QA: Bangla-English Roman
  - Summarization: BanglaCHQ-Summ
  - Reasoning: SOMADHAN (Bengali math)
- **Baselines**: Raw SLM, CoT, Uniform RAG, Chain-of-Verification - সবাইকে হারাতে হবে
- **Ablations**: ৫টা component বন্ধ করে দেখবে কোনটা সবচেয়ে important - এটাই প্রমাণ করবে তোমার method কাজ করে, dataset না

---

## কেন করছে? (Why - Motivation)

1. **বাংলা huge but low-resource**: ৩০০M+ speaker, ৬ষ্ঠ বৃহত্তম ভাষা, কিন্তু AI safety tooling নেই
2. **SLM বেশি hallucinate করে**: Paper দেখিয়েছে ছোট মডেল বড় মডেলের চেয়ে বেশি ভুল করে। আর বাংলাদেশের মতো low-resource setting এ ছোট মডেলই deploy করতে হয় (cheap)
3. **কেউ mitigation করেনি**: BenHalluEval (2026) শুধু measure করে, fix করে না। তোমার কাজ first mitigation
4. **CoT fails in Bengali**: BenHalluEval দেখিয়েছে Chain-of-Thought 21 টার মধ্যে 12 টা ক্ষেত্রে hallucination বাড়ায় - তাই নতুন method দরকার
5. **Code-mixed reality**: বাংলাদেশে মানুষ "Dhaka er population koto?" এরকম লেখে - এটার জন্য কোনো solution নেই

---

## করে লাভ কি? (Impact)

### Academic:
- **First framework** - thesis হিসেবে strong, publishable
- **Method-level novelty** - শুধু dataset change না, নতুন routing mechanism
- **Workshop paper**: BLP / LoResLM এ submit করতে পারবে
- **6 contributions** proposal এ লেখা আছে - defense এ বলতে পারবে

### Real-world:
- Medical bot ভুল ওষুধের dose দেবে না
- Summarizer মিথ্যা তথ্য যোগ করবে না
- Math tutor ভুল calculation শেখাবে না
- বাংলা speaker রা safe AI পাবে

### Career:
- RAG, NLI, Uncertainty, SLM - সব hot skill শিখবে
- Kaggle, HuggingFace, FAISS experience
- Thesis টা GitHub এ portfolio হিসেবে থাকবে

---

## কিভাবে Complete করবে? (Step-by-Step Guide)

### Phase 1: Setup (1 week) - DONE by me
- [x] Repo structure বানানো - src/, data/, docs/, notebooks/
- [x] Full pipeline code skeleton - তুমি এখনই `python run_demo.py` চালিয়ে flow দেখতে পারো
- [x] Requirements, config, documentation

### Phase 2: Data & Index (1-2 weeks)
```bash
# Kaggle এ করবে
pip install -r requirements.txt

# TyDiQA download
python src/data/loaders.py

# Wikipedia dump download (bnwiki + enwiki)
# https://dumps.wikimedia.org/bnwiki/latest/ - bnwiki-latest-pages-articles.xml.bz2
# Chunk + Embed + FAISS index build
# Code আছে src/retrieval/retriever.py এ
```

### Phase 3: RQ1 - Uncertainty Calibration (2-3 weeks) - Most Important
- Kaggle notebook: 01_uncertainty_calibration.ipynb
- Qwen2.5-1.5B দিয়ে 100 টা Bengali QA sample এ 7 বার করে generate
- Semantic Entropy, Self-Consistency, Verbalized Confidence compute
- ECE, AUROC বের করো native vs code-mixed এ
- Graph বানাও - কোন signal best?
- **এটা তোমার thesis এর first result, early করো - risk কমবে**

### Phase 4: Triage + Correction (3-4 weeks)
- Triage router train/test - rule-based দিয়ে শুরু, পরে LLM-based
- Data-driven corrector: retrieval + revision prompt
- Reasoning corrector: sub-claim + NLI check
- Individual component accuracy measure করো

### Phase 5: Full Pipeline + Baselines (2 weeks)
- Full SETU pipeline run on 4 tasks
- 4 baselines run: raw, CoT, uniform RAG, CoVe
- BenHalluScore compute
- Ablation: 5 টা component একটা একটা বন্ধ করে দেখো score কত বাড়ে

### Phase 6: Writing (2-3 weeks)
- Thesis Chapter 1-5 লেখা - structure docs/THESIS_STRUCTURE.md এ আছে
- Presentation slides - proposal presentation থেকে idea নাও
- Demo video + GitHub README update

---

## তোমার জন্য আমি কি বানিয়ে দিয়েছি?

1. **Full Codebase** (12 files):
   - `src/config.py` - সব config এক জায়গায়
   - `src/data/unified_schema.py` - 4 টা dataset এক format এ
   - `src/models/slm_generator.py` - 4-bit SLM inference
   - `src/models/nli_model.py` - multilingual NLI
   - `src/pipeline/claim_decomposer.py` - Bengali claim splitter
   - `src/pipeline/uncertainty_scorer.py` - 3 signals + ECE/AUROC
   - `src/pipeline/triage_router.py` - data vs reasoning classifier
   - `src/pipeline/data_driven_corrector.py` - RAG + cross-lingual
   - `src/pipeline/reasoning_corrector.py` - BTPROP-lite
   - `src/pipeline/abstention.py` - "আমি নিশ্চিত নই"
   - `src/pipeline/reassembly.py` - final answer
   - `src/pipeline/setu_pipeline.py` - full orchestration
   - `src/evaluation/benhallu_score.py` - dual-track metric
   - `src/main.py` - CLI entry
   - `run_demo.py` - dummy demo (এখনই চালাতে পারো)

2. **Documentation**:
   - `docs/THESIS_ROADMAP.md` - 6 months plan
   - `docs/THESIS_STRUCTURE.md` - Chapter breakdown
   - `docs/ANALYSIS_BN.md` - এই file
   - `README.md` - GitHub ready

3. **Demo**:
   - `python run_demo.py` - dummy model দিয়ে pipeline flow দেখায়
   - Real model এ চালাতে: `python src/main.py --input "তোমার প্রশ্ন"`

---

## Next Steps - তুমি এখন কি করবে?

1. **এই analysis টা supervisor কে দেখাও** - Table 1 (positioning) আর pipeline diagram বুঝিয়ে দাও
2. **Kaggle account ready করো** - T4 GPU free পাবে
3. **আমাকে বলো**:
   - কোন part টা বুঝতে সমস্যা হচ্ছে?
   - Real model দিয়ে run করতে চাও নাকি dummy দিয়ে thesis লেখা শুরু করবে?
   - Dataset download এ help লাগবে?
   - Thesis writing LaTeX এ করবে নাকি Word এ?

আমি তোমার full thesis partner - model run থেকে documentation সব help করবো।

**One-Sentence Defense Line (মুখস্থ রাখো):**
"To the best of our knowledge, SETU is the first framework to reduce (not merely detect) hallucination in Bengali small language models, using a novel uncertainty-triggered triage that routes each flagged claim to type-specific correction — cross-lingual retrieval for knowledge gaps, belief-consistency for reasoning errors — and abstains when unresolved, evaluated with dual-track BenHalluScore across native and code-mixed Bengali."

---

Good luck boss! 🚀
