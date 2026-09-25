# Kaggle e SETU Run - Zero to Hero (Ekdom Beginner er jonno)

> **Tumi ekhono kichu koro nai? Eta tomar jonno.** Ekdom 1st theke step by step.

---

## Phase 0: Tomar Repo te ki ache? (2 min)

**Already done (amar previous session e):**
- `src/` folder e 12 ta pipeline module likha ache, main branch e merged (PR #1, #2, #3)
- `notebooks/SETU_Kaggle_Full.ipynb` - Week 1 er notebook (8 cell, real model)
- `notebooks/SETU_Week2_Full.ipynb` - Week 2 er notebook (BenHalluEval + RQ1 + Baselines)
- `src/data/benhallu_eval_loader.py` - BenHalluEval 12K loader (synthetic + real)
- `src/retrieval/wiki_builder.py` - Real Bengali Wikipedia FAISS builder
- `src/evaluation/rq1_calibration.py` - RQ1 calibration (ECE/AUROC)
- `src/evaluation/run_benchmark.py` - Benchmark runner (Table 1)

**Tumar kaj:** Kaggle e notebook import kore run kora. Code likhte hobe na.

---

## Phase 1: Kaggle Account Ready (5 min)

1. **kaggle.com** e jao, account banao (Gmail diye)
2. **Phone verification** koro (na korle GPU pabe na)
3. **Settings -> Account -> Phone verification** - OTP dao
4. Verify hole **GPU quota** pabe: 30 hours/week free T4 x2

**Check:**
- Kaggle homepage -> right side -> `Settings` e gele `GPU` quota dekhabe

---

## Phase 2: GitHub Repo Public vs Private (2 min, IMPORTANT)

Tomar repo: `github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98`

**2 ta path:**

### Path A: Public koro (Recommended, 2 click)

1. GitHub repo page e jao
2. **Settings** (repo er settings, profile er na - upore Code, Issues, Pull requests er pashe Settings tab)
3. Niche scroll -> **Danger Zone** -> **Change repository visibility** -> **Change to public**
4. Repo name type kore confirm koro: `einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98`
5. Done! Ekhon Kaggle e `git clone` direct kaj korbe, kono token lagbe na

**Keno public valo?**
- Thesis er jonno portfolio hobe
- Supervisor link dekhte parbe
- Job interview e dekhaite parbe
- Code e kono secret nai (ami scan korechi)

### Path B: Private rakho, Zip method use koro

Jodi public korte na chao:
1. GitHub repo -> **Code** (green button) -> **Download ZIP**
2. File save hobe: `A-Selective-Triage-...-main.zip`
3. Kaggle e upload korbe (Phase 3 e bolbo)

**Note:** Ami agent, amar token e repo public korar permission nai (403 error). Tomake nijei 2 click e korte hobe.

---

## Phase 3: Kaggle Notebook Import (3 min)

### Week 1 er jonno:

1. **kaggle.com** -> **Create** (upore) -> **New Notebook**
2. Ekta blank notebook khulbe
3. **File** menu -> **Import Notebook** -> `SETU_Kaggle_Full.ipynb` upload koro
   - File location: tomar PC te repo theke `notebooks/SETU_Kaggle_Full.ipynb` download kore nao
   - Ba GitHub e file ta open kore **Raw** -> save as .ipynb
4. Notebook import hoye gele, **right side panel** e dekho:
   - **Accelerator** -> `GPU T4 x2` select koro
   - **Internet** -> `ON` koro (MUST! OFF thakle `Could not resolve host: github.com` error asbe)
5. **Save** button e click koro (Ctrl+S)

### Jodi Path B (Zip) use koro:

1. Same notebook e, right panel e **Add Input** -> **Upload** -> zip file upload koro
2. Upload hole `/kaggle/input/` e zip thakbe
3. Cell 1 nijei auto detect kore unzip kore nebe

---

## Phase 4: Cell by Cell Run - Week 1 (15-20 min)

**Order mante hobe: Cell 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7**

### Cell 1: Setup

```python
!pip install -q transformers accelerate ...
!git clone https://github.com/einadid/...
```

- **Ki kore:** pip install + repo clone (public hole) ba zip theke extract (private hole)
- **Success check:** Last line e `Repo root: /kaggle/working/A-Selective-...` + `src/` list dekhabe
- **Fail hole:**
  - `Could not resolve host: github.com` -> Internet OFF, ON koro (right panel)
  - `Authentication failed` -> repo private, Path B (zip) use koro ba public koro
  - `Repo setup fail` -> zip upload koro nai, Add Input theke upload koro

### Cell 2: Import + GPU Check

- **Ki kore:** 13 ta SETU module import + GPU check
- **Success:** `All SETU modules imported OK` + `GPU 0: Tesla T4 | 15.8 GB`
- **Fail:** `No module named 'transformers'` -> Cell 1 abar run koro

### Cell 3: Real Model - Qwen2.5-1.5B 4-bit

- **Ki kore:** Qwen model load (4-bit quantization) + 3 ta test question
- **Time:** First time 4-6 min (model download ~3GB)
- **Success:** 3 ta Q&A er answer ber hobe:
  - `বাংলাদেশের রাজধানী কোথায়?` -> `ঢাকা` (correct)
  - `ঢাকার জনসংখ্যা...` -> sometimes `৫ কোটি` (hallucination, thesis er example)
  - `Dhaka city te koto lok...` -> code-mixed
- **Fail:**
  - `CUDA out of memory` -> model change koro: `Qwen/Qwen2.5-0.5B-Instruct` (Cell 3 e first line e)
  - Stuck -> wait koro, model download hocche

### Cell 4: Retriever + Demo Corpus

- **Ki kore:** BGE-M3 embedder load (~2GB) + demo corpus (7 ta sentence) + FAISS index + cross-lingual test
- **Time:** 3-5 min first time
- **Success:** `Q: ঢাকার জনসংখ্যা কত? | fallback: False | max_score: 0.85` etc
- **Note:** Eta demo corpus, real Wikipedia Week 2 e

### Cell 5: Decomposition + Uncertainty + Triage

- **Ki kore:** Draft -> atomic claims -> triage (data-driven vs reasoning-driven) + uncertainty (3 signals)
- **Time:** 2-4 min
- **Success check (IMPORTANT - thesis er main contribution):**
  ```
  [1] ঢাকার জনসংখ্যা প্রায় ১ কোটি ৫০ লাখ।
      TRIAGE -> data-driven (conf=0.65, data=1, reasoning=0)
      UNCERT -> combined=0.72 flagged=True
  [2] ৫ + ৩ = ৯
      TRIAGE -> reasoning-driven (conf=0.80, data=0, reasoning=2)
  ```
  - Population-type claim -> data-driven (RAG e jabe)
  - Math-type claim -> reasoning-driven (NLI check e jabe)
  - Eta kaj korle thesis er core idea proven

### Cell 6: FULL SETU Pipeline End-to-End

- **Ki kore:** Full pipeline: draft -> claims -> triage -> correction (RAG or belief-consistency) -> abstention -> reassembly
- **Time:** 2-4 min
- **Success:**
  ```
  FINAL ANSWER: ঢাকার জনসংখ্যা প্রায় ১ কোটি ৪৫ লাখ (২০২২ আদমশুমারি অনুযায়ী)...
  STATS: {"n_claims": 4, "n_flagged": 2, "n_data_driven": 1, "n_reasoning_driven": 1, "n_corrected": 1, "n_abstained": 0}
  ```
  - FINAL ANSWER e corrected info thakbe
  - STATS e numbers

### Cell 7: Supervisor Report Generate

- **Ki kore:** `/kaggle/working/SUPERVISOR_REPORT_week1.md` file banay
- **Success:** File saved message + report text dekhabe
- **Download:** Right panel e **Output** tab -> `SUPERVISOR_REPORT_week1.md` -> download icon

---

## Phase 5: Screenshot + Supervisor ke Pathano (5 min)

**3 ta screenshot lagbe:**

1. **Cell 2:** GPU + `All modules imported OK`
2. **Cell 5:** Triage labels (data-driven vs reasoning-driven)
3. **Cell 6:** FINAL ANSWER + STATS

**Kivabe screenshot:**
- Cell er output er upore mouse nile camera icon asbe, ba Windows e `Win+Shift+S`

**Supervisor ke ki pathabe:**

1. 3 ta screenshot
2. `SUPERVISOR_REPORT_week1.md` file
3. `docs/SUPERVISOR_UPDATE_WEEK1.md` er text copy-paste (email e)

**Email template:**
```
Subject: SETU Week 1 Update - Pipeline Working on Kaggle

Sir,

Week 1 completed:

- Full framework implemented (12 modules) - code on GitHub main branch
- Kaggle T4 x2 GPU te Qwen2.5-1.5B 4-bit running
- Tested 3 questions: control (correct), numeric (hallucination prone), code-mixed
- Triage router working: population -> data-driven (0.65), math 5+3=9 -> reasoning-driven (0.80) - main contribution validated
- Full end-to-end pipeline: draft -> claims -> triage -> correction -> final answer
- Report attached + screenshots

Next: Week 2 - BenHalluEval 12K loader, real Wikipedia FAISS, RQ1 calibration, baselines

GitHub: https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98

Regards,
Kazi & Raihan
```

---

## Phase 6: Week 2 Notebook Run (1-1.5 hour)

Week 1 er moto same steps, but notebook different:

1. **kaggle.com -> Create -> New Notebook**
2. **Accelerator = GPU T4 x2, Internet = ON**
3. **File -> Import Notebook -> `SETU_Week2_Full.ipynb`**
4. **Cell 1 run:** setup (same)
5. **Cell 2 run:** import (including Week2 new modules)
6. **Cell 3 run:** BenHalluEval loader
   - `max_samples_per_task=20` = quick test (~220 instances, 2 min)
   - `max_samples_per_task=100` = real experiment (~1100 instances, 5 min)
   - Output: stats - total, by task, by hallucination type
7. **Cell 4 run:** Wikipedia FAISS
   - `MODE="demo"` = 300 passages, 1 min (quick)
   - `MODE="hf_500"` = 500 articles ~5k passages, 20-30 min (recommended for thesis)
   - `MODE="hf_5000"` = 5000 articles ~50k passages, 2-3 hours (full)
   - Output: index stats - n_vectors, n_passages
8. **Cell 5 run:** RQ1 Calibration + Real Model
   - Loads Qwen + NLI (4-6 min)
   - RQ1: 50 claims = ~10-15 min, 200 claims = ~1 hour
   - Output: Table | Signal | AUROC | ECE |
   - Expected: Semantic entropy best AUROC, code-mixed worse ECE
9. **Cell 6 run:** Benchmark (Baselines + SETU)
   - 30 samples per method = ~30 min, 100 samples = ~1.5 hours
   - Output: Table 1 BenHalluScore
   - Expected: SETU best, CoT increases false alarm (matches BenHalluEval paper)
10. **Cell 7 run:** Reports generate
    - `SUPERVISOR_REPORT_week2.md` + `rq1_calibration.md` + `benchmark_results.md`
    - Download from Output tab

**Screenshot for Week 2:**
- Cell 3: BenHalluEval stats
- Cell 4: FAISS index stats
- Cell 5: RQ1 table
- Cell 6: Benchmark Table 1

---

## Phase 7: Troubleshooting (Common Errors)

| Error | Keno hoy | Fix |
|---|---|---|
| `Could not resolve host: github.com` | Internet OFF | Right panel -> Internet ON -> Save -> Run again |
| `Authentication failed` | Repo private + git clone | Path B zip use koro, ba repo public koro (Settings -> Danger Zone) |
| `Repo setup fail` | Zip upload koro nai | Add Input -> Upload Dataset -> zip upload |
| `CUDA out of memory` | 1.5B model boro, T4 VRAM 16GB but 2 GPU share | Cell 3 e `Qwen/Qwen2.5-0.5B-Instruct` koro |
| `No module named 'faiss'` | pip install fail | Cell 1 abar run koro, Internet ON check |
| Cell 3 stuck 10+ min | Model download hocche (~3GB) | Wait koro, Kaggle internet slow hote pare |
| Cell 4 slow | BGE-M3 download (~2GB) | Wait koro, first time only |
| Cell 5 `Triage failed` | SLM None | Cell 3 success kina check, gen variable ache kina |
| Session crash | GPU memory full | Runtime -> Restart, abar Cell 1 theke run |

**Amake ki pathabe jodi atke jao:**
- Screenshot (error message soho)
- Cell number (Cell 3? 4?)
- Kaggle Settings screenshot (GPU + Internet ON kina)

---

## Phase 8: GitHub e Code Save (Auto)

Tumi Kaggle e run korle code change hoy na, main repo te already code ache. Kintu jodi tumi local e kichu change koro:

```bash
git add .
git commit -m "your message"
git push origin arena/01a0d2e6-a-selective-triage-and-correct
```

Ami already Week2 er code push kore diyechi, tumi pull korlei pabe.

---

## Summary Checklist (Zero to Hero)

- [ ] Kaggle account + phone verification
- [ ] GitHub repo public (Path A) ba zip download (Path B)
- [ ] Kaggle -> New Notebook -> Import `SETU_Kaggle_Full.ipynb`
- [ ] Right panel: GPU T4 x2 + Internet ON
- [ ] Cell 1-7 run (Week1) - 15-20 min
- [ ] 3 screenshot + report download
- [ ] Supervisor ke email
- [ ] Week2: Import `SETU_Week2_Full.ipynb` + same settings
- [ ] Cell 1-7 run (Week2) - 1-1.5 hour
- [ ] 4 screenshot + 3 report download
- [ ] Supervisor ke Week2 email

**Total time:** Week1 30 min, Week2 2 hours (including waiting)

---

Good luck! Kono step e atkle amake screenshot dao.
