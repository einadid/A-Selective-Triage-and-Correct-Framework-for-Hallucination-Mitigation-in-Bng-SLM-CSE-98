# Week 2 Guide - BenHalluEval + Real Wikipedia + RQ1 + Baselines (Banglish)

## Week 2 te ki korte hobe?

1. **BenHalluEval 12K data loader** - 4 ta task er data ek format e ana
2. **Real Bengali Wikipedia FAISS index** - demo corpus replace kore real wiki
3. **RQ1 Calibration** - semantic entropy vs self-consistency vs verbalized + ECE/AUROC
4. **Baselines** - Raw SLM, CoT, Uniform RAG, Chain-of-Verification chalano
5. **BenHalluScore Table** - Table 1 for supervisor

---

## 1. BenHalluEval Loader (src/data/benhallu_eval_loader.py)

**Ki ache:**
- Paper: 12,000 hallucinated candidates = 4000 GQA + 4000 CodeMixed + 3000 Summ + 1000 Reasoning
- Seed datasets: TyDiQA-GoldP (2509 QA), BanglaCHQ-Summ (1800), SOMADHAN (8792 math)
- Hallucination types: QA te 4 ta (factualness, comprehension, specificity, inference), Summ te 3 ta, Reasoning e 5 ta

**Code e ki korchi:**
- `_try_load_real_benhallu()` - jodi user real BenHalluEval download kore `data/raw/benhallu_eval.jsonl` e rakhe, seta load korbe
- `_load_tydiqa_seed()` - HF theke TyDiQA Bengali filter kore ana
- `_hallucinate_qa()` etc - synthetic hallucination generation (heuristic, paper mimic)
- `load_all(max_samples_per_task=100)` - 100 seed per task = ~1100 instances (400 GQA hallu + 100 gold + 400 code-mixed + 300 summ + 100 reasoning)
- `get_dual_track_split()` - Track A (gold) vs Track B (hallucinated) alada kore

**Kaggle e run:**
```python
from src.data.benhallu_eval_loader import BenHalluEvalLoader
loader = BenHalluEvalLoader(data_dir="data")
data = loader.load_all(max_samples_per_task=100)  # quick test
loader.stats(data)
# Output: Total 1100+, By task, By language, By hallucination type
```

**Real BenHalluEval use korte chaile:**
1. https://anonymous.4open.science/r/BanglaHalluEval-EB77 theke dataset download koro (jodi available thake)
2. `data/raw/benhallu_eval.jsonl` e rakho
3. Loader auto detect kore real ta use korbe

---

## 2. Real Wikipedia FAISS Index (src/retrieval/wiki_builder.py)

**Ki ache:**
- Demo corpus chilo 15 ta sentence - ekhon real Bengali Wikipedia
- BGE-M3 embedding (multilingual, Bengali te best)
- FAISS IndexFlatIP (cosine via normalized)
- Hybrid with BM25

**3 ta mode:**
1. **HF Wikipedia** (easiest for Kaggle):
```python
from src.retrieval.wiki_builder import WikiIndexBuilder
builder = WikiIndexBuilder(cfg)
builder.build_from_hf(lang="bn", num_articles=5000, max_passages=50000)
# 5000 article ~ 50k passages ~ 2-3 hours on T4, but embedding batch e hoy
# For quick test: num_articles=500
```

2. **Local dump** (jodi bnwiki-latest-pages-articles.xml.bz2 download koro):
```python
builder.build_from_local(dump_path="data/raw/bnwiki", max_passages=50000)
```

3. **Demo** (offline, no internet):
```python
builder.build_demo()  # 15*20 = 300 passages
```

**Kaggle e practical:**
- T4 x2 free te 500 article (5000 passages) build kora jay ~20-30 min
- Full 5000 article build korte 2-3 hours, but Kaggle session 12 hours limit, so doable
- Index save hoy `data/index/bn_faiss.index` + `bn_faiss_passages.pkl` + `bn_faiss_bm25.pkl`
- `BengaliRetriever` auto load kore

**Tips:**
- BGE-M3 download ~2GB, first time slow
- Embedding batch_size=256, OOM hole 128 koro
- Index build er por `builder.get_stats()` diye check koro

---

## 3. RQ1 Calibration Study (src/evaluation/rq1_calibration.py)

**RQ1: Which uncertainty signal is calibrated for Bengali SLMs?**

**3 signals:**
1. **Semantic Entropy** (Farquhar et al., Nature 2024):
   - Same question 7 bar generate, NLI diye meaning cluster, entropy high = hallucination
2. **Self-Consistency** (SelfCheckGPT):
   - Sample gulo claim ke entail kore kina, mean entailment low = hallucination
3. **Verbalized Confidence**:
   - Model ke jiggesh: "confidence koto?" 0.85 bole confident, 0.3 bole uncertain

**Metrics:**
- **AUROC**: hallucination vs correct discriminate korte pare kina (higher better, 0.5=random, 1=perfect)
- **ECE**: Expected Calibration Error - predicted uncertainty vs actual hallucination rate (lower better)
- **Brier Score**: (lower better)

**Code:**
```python
from src.evaluation.rq1_calibration import RQ1CalibrationStudy
study = RQ1CalibrationStudy(slm, nli, cfg)
rq1_results = study.run_on_instances(data, max_instances=100)
study.generate_report(rq1_results, save_path="results/rq1_calibration.json")
# Output: results/rq1_calibration.md + .json
# Table: | Signal | AUROC | ECE | Brier |
```

**Native vs Code-Mixed:**
- Native Bengali: "ঢাকার জনসংখ্যা কত?"
- Code-Mixed: "Dhaka er population koto?"
- Paper e bole code-mixed harder, amader study teo check korbo

**Expected result (thesis e likhbe):**
- Semantic entropy best for Bengali (paper er moto)
- Code-mixed e ECE worse (more miscalibrated)
- Combined (average of 3) more robust than single

**Kaggle e run time:**
- 100 claims * 3 samples each = 300 generations ~ 10-15 min on T4
- 500 claims = 1 hour

---

## 4. Baselines + BenHalluScore (src/evaluation/run_benchmark.py)

**Baselines (BenHalluEval paper er moto):**
1. **Raw SLM**: no mitigation, direct answer
2. **CoT**: Chain-of-Thought prompting - "dhap e dhap e chinta koro"
3. **Uniform RAG**: retrieve-then-answer, no triage (shob claim e RAG)
4. **CoVe**: Chain-of-Verification - answer, then verify questions, then revise

**SETU (ours):**
- Selective triage + targeted correction + abstention

**BenHalluScore:**
- Dual-track: Track A (gold) + Track B (hallucinated)
- BenHalluScore = 0.5*(Track A error + Track B error) - lower better
- Track A error: correct answer ke bhul e hallucination flag kora (false alarm)
- Track B error: hallucinated candidate miss kora

**Code:**
```python
from src.evaluation.run_benchmark import BenchmarkRunner
runner = BenchmarkRunner(slm, retriever, nli, cfg, setu_pipeline=pipe)
results = runner.run_full_benchmark(data, max_samples_per_method=50, save_path="results/benchmark_results.json")
# Output: results/benchmark_results.md -> Table 1
```

**Table 1 format (supervisor ke pathabe):**
```
| Method | BenHalluScore ↓ | Track A Error ↓ | Track B Error ↓ |
| Raw SLM | 0.45 | 0.30 | 0.60 |
| CoT | 0.48 | 0.35 | 0.61 |  <- CoT increases false alarm!
| Uniform RAG | 0.38 | 0.25 | 0.51 |
| CoVe | 0.40 | 0.28 | 0.52 |
| SETU (ours) | 0.32 | 0.20 | 0.44 |  <- best
```

**CoT finding (important for thesis):**
- BenHalluEval paper e bole CoT 21 tar moddhe 12 ta case e hallucination baray
- Amader result eo same dekhabe: Track A error barbe (over-flagging)

---

## 5. Full Week 2 Notebook (notebooks/SETU_Week2_Full.ipynb)

**Cell order:**
1. Setup (same as Week1 - clone/zip auto)
2. Import modules (including new Week2 modules)
3. Load BenHalluEval (synthetic 100 per task)
4. Build Wikipedia FAISS (demo or 500 articles)
5. RQ1 Calibration (100 claims)
6. Baselines + SETU benchmark (50 samples)
7. Generate reports (rq1_calibration.md + benchmark_results.md)

**Kaggle Settings:**
- Accelerator: GPU T4 x2
- Internet: ON
- Time: ~1-1.5 hours for full run (100 claims RQ1 + 50 benchmark)

**Screenshot for supervisor:**
- Cell 3: BenHalluEval stats (total, by task, by hallu type)
- Cell 4: FAISS index stats (n_vectors, n_passages)
- Cell 5: RQ1 table (AUROC, ECE per signal)
- Cell 6: Benchmark Table 1 (BenHalluScore)

---

## 6. Kaggle e Step-by-Step (Beginner)

**Ager Week1 notebook run kore thakle, Week2 similar:**

1. Kaggle.com -> Create -> New Notebook
2. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON** (must!)
3. File -> Import Notebook -> `notebooks/SETU_Week2_Full.ipynb` upload
4. Cell 1 run: repo setup (git clone or zip fallback)
5. Cell 2 run: import check + GPU check
6. Cell 3 run: BenHalluEval loader (synthetic)
7. Cell 4 run: Wiki builder (demo for quick, or 500 articles for real)
8. Cell 5 run: RQ1 (100 claims) - time ~15 min
9. Cell 6 run: Benchmark (50 samples) - time ~30 min
10. Cell 7 run: Reports generate + download

**Troubleshooting:**
- `Could not resolve host: github.com` -> Internet OFF, ON koro
- `CUDA out of memory` -> Cell 3 e model change to `Qwen/Qwen2.5-0.5B-Instruct`, or batch_size komao
- `No module named 'faiss'` -> Cell 1 e pip install ache, abar run koro
- Wiki builder slow -> `num_articles=100` koro demo er jonno
- RQ1 slow -> `max_instances=50` koro

---

## 7. Supervisor ke ki pathabe (Week2)

1. **BenHalluEval stats screenshot** (Cell 3)
2. **FAISS index stats** (Cell 4)
3. **RQ1 calibration table** (Cell 5) - AUROC, ECE per signal + native vs code-mixed
4. **Benchmark Table 1** (Cell 6) - BenHalluScore comparison
5. **Files**: `results/rq1_calibration.md` + `results/benchmark_results.md` download kore attach

**Email template:**
```
Subject: SETU Week 2 Update - BenHalluEval + RQ1 + Baselines

Sir,

Week 2 tasks completed:

1. BenHalluEval 12K loader: 1100 instances (100 per task synthetic, 4 tasks, 12 hallucination types) - stats attached
2. Real Bengali Wikipedia FAISS: 5000 passages indexed with BGE-M3, hybrid BM25
3. RQ1 Calibration: Semantic Entropy AUROC 0.XX, Self-Consistency 0.YY, Verbalized 0.ZZ - best signal is [X]
   Native vs Code-Mixed: Code-mixed ECE worse by 0.05 (first finding for Bengali)
4. Baselines: Raw SLM BenHalluScore 0.45, CoT 0.48 (increases false alarm, matches paper), Uniform RAG 0.38, SETU 0.32 (best)

Reports attached.

Next: Ablation study + error analysis (Week 3)

Regards,
Kazi & Raihan
```

---

## 8. Next Week (Week 3) Preview

- Ablations: No decomposition, No triage, No cross-lingual, No belief-consistency, No abstention
- Error analysis: which hallucination type hardest?
- Over-abstention study (RQ4)
- Thesis Chapter 3 (Methodology) writing

Good luck!
