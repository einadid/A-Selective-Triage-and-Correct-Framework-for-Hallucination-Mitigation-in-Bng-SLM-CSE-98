# Kaggle e SETU Run - Final Guide (2 ta path, jeta tomake Shohoj)

**Notebook:** `notebooks/SETU_Kaggle_Full.ipynb` — ei ta import koro Kaggle e, tarpor shob cell run.
**Cell 1 nijei repo setup kore** - git clone chesta kore, fail korle `/kaggle/input` er zip theke ney.
Mane duto path e same notebook e kaj korbe.

---

## Kaggle Settings (Age Ei Kaj Ta Koro - Must)

Kaggle notebook er **right side panel** e:
- **Accelerator** -> `GPU T4 x2`
- **Internet** -> `ON`   *(off thakle `Could not resolve host: github.com` error asbe)*

---

## Path A: Repo Public koro (best - ekbar kore felo)

1. GitHub repo page -> **Settings** (repo er settings, profile na)
2. Nicher dike scroll -> **Danger Zone** -> **Change repository visibility** -> **Change to public**
3. Confirm likhe dao repo er naam
4. Ekhon Kaggle e notebook Cell 1 run korlei hobe — token, zip kichui lagbe na

> Thesis er jonno public kora valo: supervisor dekhbe, examiner dekhbe, portfolio hobe,
> ar dissertation e link dewa jabe. Code er moddhe kono secret/token nai (ami scan korechi).

---

## Path B: Private rekhe Zip method (ekhon i kaj korbe, kono permission lagbe na)

1. GitHub repo -> **Code** (green button) -> **Download ZIP**
   - File name: `A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98-main.zip`
   - (Ami ekhane ekta ready zip o diyechi: `setu-code-for-kaggle.zip`)
2. Kaggle notebook -> right panel -> **Add Input** -> **Upload Dataset** -> zip ta upload koro
3. Notebook er **Cell 1** run koro — code ta zip ta khuje ber kore nijei extract kore nebe

GitHub link e jete na chaile: ami `setu-code-for-kaggle.zip` ta diye diyechi — sudhu download kore upload koro.

---

## Run Order (shob cell shift+enter)

| Cell | Ki kore | Time |
|---|---|---|
| 1 | pip install + repo setup (clone/zip auto) | 1-2 min |
| 2 | 13 ta SETU module import + GPU check | 30 sec |
| 3 | Qwen2.5-1.5B 4-bit load + 3 ta test question (control / numeric / code-mixed) | 4-6 min (first time) |
| 4 | BGE-M3 retriever + demo corpus + FAISS index + cross-lingual test | 3-5 min |
| 5 | Draft -> claim decomposition -> uncertainty (RQ1) -> triage | 2-4 min |
| 6 | **Full SETU pipeline end-to-end** (draft -> claims -> triage -> correction -> abstention -> reassembly) | 2-4 min |
| 7 | Supervisor report auto-generate (`/kaggle/working/SUPERVISOR_REPORT_week1.md`) | 5 sec |

**Total ~15-20 min** — T4 x2 free tier te fit kore.

---

## Success Check (kivabe jaanbe shob thik)

- Cell 1: `Repo root : /kaggle/working/A-Selective-...` + `src/` list dekhay
- Cell 2: `All SETU modules imported OK` + GPU name + VRAM
- Cell 3: Bengali answer gulo ber hoy (kono blank na)
- Cell 5: `TRIAGE -> data-driven` / `reasoning-driven` label + uncertainty number
- Cell 6: `FINAL ANSWER` er por `STATS` JSON (n_claims, n_flagged, n_data_driven, n_abstained)
- Cell 7: report file — **Output** tab theke download koro

**Screenshot lagbe 3 ta:** Cell 2 (GPU + modules), Cell 5 (triage labels), Cell 6 (FINAL ANSWER + stats)।
Ei 3 ta supervisor ke pathao.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not resolve host: github.com` | Internet **OFF** ache -> Settings e ON koro |
| `Authentication failed` / `could not read Username` | repo private + Path B (zip) use koro, ba repo public koro |
| `CUDA out of memory` | Cell 3 e model change: `Qwen/Qwen2.5-0.5B-Instruct` |
| Cell 4 slow / stuck | BGE-M3 (~2GB) download hocche, wait koro |
| Cell 5 time lagche | `cfg.uncertainty.n_samples = 3` ache; 1 o korte paro |
| Session e GPU nei | Settings -> Accelerator -> GPU T4 x2 -> Save -> session restart |

---

## Er Pore Ki (Week 2)

1. `BenHalluEval` 12K dataset load + unified schema (`src/data/loaders.py`)
2. Real Bengali Wikipedia diye FAISS index build (`ret.build_index_from_wiki_dump`)
3. RQ1 full experiment: semantic entropy vs self-consistency vs verbalized confidence + ECE/AUROC
4. Baselines chalano: Raw SLM, CoT, Uniform RAG, Chain-of-Verification (`src/evaluation/baselines.py`)
5. BenHalluScore diye comparison table (Table 1 - supervisor approval lagbe)
