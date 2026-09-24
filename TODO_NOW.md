# SETU - Ekhon Ki Korba (Status + Next Steps)

**Last update: 2026-09-24**

---

## ✅ Ja hoye geche (done)

| Kaj | Status |
|---|---|
| GitHub repo te full code | ✅ **main branch e ache** (PR #1, #2 merged) |
| 12 ta pipeline module written | ✅ `src/pipeline/`, `src/retrieval/`, `src/models/`, `src/evaluation/` |
| Core logic test (CPU) | ✅ `docs/VERIFIED_PIPELINE_TEST.md` - real output |
| Kaggle notebook ready | ✅ `notebooks/SETU_Kaggle_Full.ipynb` (8 cell, real model) |
| Kaggle guide | ✅ `notebooks/KAGGLE_SETUP_GUIDE.md` |
| Supervisor Week-1 report | ✅ `docs/SUPERVISOR_UPDATE_WEEK1.md` |
| Zip file (Kaggle upload er jonno) | ✅ `setu-code-for-kaggle.zip` |

**Verified test result (real, banano na):**
- Claim decomposition: 1 ta draft -> 4 ta atomic claim
- Triage: `ঢাকার জনসংখ্যা ১ কোটি ৫০ লাখ` -> **data-driven** (0.65)
- Triage: `৫ + ৩ = ৯` -> **reasoning-driven** (0.80)   ← thesis er main contribution kaj korche
- Abstention: `আমি নিশ্চিত নই` phrase thik moto ashche

---

## 🔴 Ekhon tomar 2 ta kaj (ajkei, ~15 min)

### Kaj 1: Kaggle e code tulte hobe — duitar ekta path
- **Path A:** GitHub repo public koro -> Settings -> Danger Zone -> Change visibility -> public
  - Er por notebook Cell 1 (git clone) direct kaj korbe
- **Path B:** Repo private rakhbe -> `setu-code-for-kaggle.zip` download koro ->
  Kaggle -> Add Input -> Upload Dataset -> zip upload -> Cell 1 nijei extract kore nebe

*(Ami repo public korte parchi na - amar GitHub token e repo-admin permission nei. 403 error ashe. Oita tomar account er 2 click er kaj.)*

### Kaj 2: Kaggle notebook run koro
1. kaggle.com -> Create -> New Notebook -> **File -> Import Notebook** -> `SETU_Kaggle_Full.ipynb` upload
2. Right panel: **Accelerator = GPU T4 x2**, **Internet = ON**
3. Cell 1 theke shob cell ek ek kore run koro (order maante hobe)
4. **3 ta screenshot nao:** Cell 2 (GPU + modules), Cell 5 (triage labels), Cell 6 (FINAL ANSWER + stats)
5. Cell 7 report file download koro

---

## 📤 Tarpor (aj rat / kal shokale)

1. Screenshot 3 ta + `SUPERVISOR_REPORT_week1.md` -> supervisor ke pathao
   (`docs/SUPERVISOR_UPDATE_WEEK1.md` er text ta email/message e paste korle hobe)
2. Amake bolo output ki aslo — ami porer step (BenHalluEval data + real FAISS index) egiye niye jabo

---

## 📅 Week 2 Plan (ami ready rakhbo)

1. `BenHalluEval` 12K dataset -> `src/data/loaders.py` + unified schema
2. Real Bengali Wikipedia theke FAISS index build (demo corpus replace)
3. **RQ1 full experiment:** semantic entropy vs self-consistency vs verbalized confidence + ECE / AUROC
4. Baselines chalano: Raw SLM, CoT, Uniform RAG, Chain-of-Verification
5. BenHalluScore diye Table 1 (supervisor approval lagbe)

---

## ❓ Jodi atke jao

| Problem | Fix |
|---|---|
| `Could not resolve host: github.com` | Kaggle Settings -> Internet **ON** |
| `Authentication failed` | repo private -> Path B (zip) use koro |
| `CUDA out of memory` | Cell 3 e `Qwen/Qwen2.5-0.5B-Instruct` koro |
| Cell 1 e `Repo setup fail` | zip upload koro nai — Add Input theke upload koro |
| Onno kichu | screenshot/error text ta amake pathao |

---

## Link

- Repo: github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
- PR #1 (merged): full framework
- PR #2 (merged): Kaggle notebook fix + verified test
