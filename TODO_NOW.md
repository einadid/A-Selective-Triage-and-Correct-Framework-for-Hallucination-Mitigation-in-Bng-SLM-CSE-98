# Ekhon Tomar Koronio Ki? - Aste Aste Checklist

## TODAY (Ajke - 30 min)

### Step 0: Repo ta bujhe nao
- [ ] Ami je file gulo baniyechi segulo ekbar khule dekho:
  - `docs/ANALYSIS_BN.md` - full analysis Banglay (eta age poro)
  - `README.md` - project overview
  - `run_demo.py` - dummy demo

### Step 1: Demo Run Koro (5 min)
Tumi ekhoni dekhte parbe pipeline kivabe kaj kore, kono GPU lagbe na:

```bash
cd /home/user/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
pip install numpy scikit-learn --break-system-packages -q
python run_demo.py
```

Output e dekhbe:
- Query -> Draft Answer
- 4 ta claim e vag
- Uncertainty scoring
- Final Answer

**Eta tomar supervisor ke dekhanor jonno perfect - je pipeline ready.**

---

## TOMORROW (Kalke - 1 hour)

### Step 2: Kaggle Setup
1. kaggle.com e account khulo (jodi na thake)
2. New Notebook -> T4 x2 GPU select koro
3. Ekhane ami ekta notebook baniye debo - tumi shudhu copy-paste korbe

**Amake bolo - tomar Kaggle ache? Ami ekhoni Kaggle ready notebook baniye debo?**

### Step 3: Paper Gulo Poro (Important)
Proposal e 6 ta base paper ache - egulo na porle defense e atkabe:

1. **BenHalluEval** (arXiv 2605.31483) - evaluation protocol, BenHalluScore formula
2. **HalluGuard** (ICLR 2026) - data-driven vs reasoning-driven taxonomy
3. **BTPROP** (NAACL 2025) - belief tree idea
4. **HalluSearch** (SemEval 2025) - atomic claim + retrieval
5. **SelfCheckGPT** (EMNLP 2023) - self-consistency signal
6. **Semantic Entropy** (Nature 2024) - entropy signal

**Shuru te 2 ta poro:** BenHalluEval + SelfCheckGPT - 1 din e hoye jabe.

---

## THIS WEEK (Ei soptahe)

### Step 4: Dataset Download (Kaggle e)
Kaggle notebook e ei code run korbe:

```python
!pip install datasets
from datasets import load_dataset

# TyDiQA Bengali - 5000 samples
ds = load_dataset("google-research-datasets/tydiqa", "secondary_task")
# Filter Bengali - ami loader code diyechi src/data/loaders.py te
```

Ar 2 ta dataset:
- BanglaCHQ-Summ: https://github.com/alvi-khan/BanglaCHQ-Summ
- SOMADHAN: Mendeley link - ami alternative dummy diyechi

**Target:** `data/processed/` folder e 3 ta JSONL file ready

### Step 5: Wikipedia FAISS Index (Optional but important)
- bnwiki dump download: https://dumps.wikimedia.org/bnwiki/latest/
- BGE-M3 embedding diye FAISS index build
- Code ready ache `src/retrieval/retriever.py` te
- Na parle ami dummy index diye demo chaliye debo, pore real banabe

---

## NEXT WEEK (Samner soptahe)

### Step 6: RQ1 - Uncertainty Calibration Study (Thesis er heart)
Eta sobcheye important, eta diye thesis er 50% hoye jabe:

Kaggle e Qwen2.5-1.5B model load korbe 4-bit e:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# 4-bit config
# Generate 7 samples per query
# Compute semantic entropy, self-consistency
```

**Output:** Graph - kon signal Banglay best? Native vs Code-mixed difference?

**Eta hole supervisor khushi, karon eta publishable finding.**

---

## TOMAR 3 TA OPTION - KONTA diye shuru korbe?

**Option A: Fast Demo Track** (Supervisor ke dekhanor jonno)
- Dummy model diye full pipeline run
- Presentation ready in 2 days
- Real model pore

**Option B: Real Model Track** (Thesis er jonno)
- Kaggle e Qwen2.5-1.5B diye real run
- 1 week lagbe but real result

**Option C: Writing Track**
- Age thesis Chapter 2 (Literature) lekha shuru
- Ami LaTeX template baniye debo

**Amake bolo kon option e jabe? Ami sei hisabe next file baniye debo.**

---

## Ami Ki Help Korbo?

- [x] Code skeleton - DONE
- [ ] Kaggle notebook (real model) - tomar answer er upor banabo
- [ ] Thesis LaTeX template - chao?
- [ ] Presentation PPT - chao?
- [ ] Dataset download e live help

**Ekhon bolo boss - Option A, B, C konta? Ar Kaggle ache?**
