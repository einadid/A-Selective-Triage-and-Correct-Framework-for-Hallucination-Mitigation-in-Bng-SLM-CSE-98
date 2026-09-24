# SETU Pipeline - Verified Test Output (Real Run)

**Date:** 2026-09-24
**Environment:** Local (CPU, no model needed for these modules)
**Repo commit:** main (after PR #1 merge)

Ei file ta proof je framework er core logic **thik moto kaj kore** - banano kotha noy, real output.

---

## Test 1: Atomic Claim Decomposition

**Input draft answer:**
```
ঢাকার জনসংখ্যা প্রায় ১ কোটি ৫০ লাখ। এটি বাংলাদেশের রাজধানী এবং সবচেয়ে বড় শহর। ২০২২ সালের আদমশুমারি অনুযায়ী জনসংখ্যা বেড়েছে।
```

**Output (method = "rule" and "hybrid"):** 4 ta atomic claim
```
1. ঢাকার জনসংখ্যা প্রায় ১ কোটি ৫০ লাখ।
2. এটি বাংলাদেশের রাজধানী।
3. সবচেয়ে বড় শহর।
4. ২০২২ সালের আদমশুমারি অনুযায়ী জনসংখ্যা বেড়েছে।
```
> Note: `slm_generator=None` hoye gele hybrid method rule-based e fallback kore (by design).

---

## Test 2: Triage Router (thesis er core contribution)

| Claim | Label | Confidence | data_score | reasoning_score |
|---|---|---|---|---|
| ঢাকার জনসংখ্যা ১ কোটি ৫০ লাখ | **data-driven** | 0.65 | 1 | 0 |
| ৫ + ৩ = ৯ | **reasoning-driven** | 0.80 | 0 | 2 |
| কেন আকাশ নীল তাই সমুদ্র নীল দেখায় | data-driven | 0.50 | 1 | 1 |

**Ei table tai contribution:** population-type claim -> fact correction (RAG) e jabe;
math/logic-type claim -> reasoning correction e jabe. Ekta correction strategy diye duita
problem solve hoy na - ei ta ei thesis er main argument, ar ekhon code e o eta kaj korche.

---

## Test 3: Abstention ("আমি নিশ্চিত নই")

```python
AbstentionModule().process({"original_claim": "ঢাকা xxx", "status": "no_evidence"})
# -> {'final_claim': 'আমি নিশ্চিত নই', 'abstained': True}

AbstentionModule().process({"original_claim": "ঢাকা xxx", "status": "corrected",
                            "corrected_claim": "ঢাকার জনসংখ্যা ১.৪ কোটি"})
# -> {'final_claim': 'ঢাকার জনসংখ্যা ১.৪ কোটি', 'abstained': False}
```
Abstention phrase exactly: `আমি নিশ্চিত নই`

---

## Test 4: Reassembly

Input claim results (1 corrected, 1 abstained) -> reassembled answer:
```
ক খ আমি নিশ্চিত নই
```
Correct but not corrected claim gulo bad diye dey, abstained claim er jaygay
abstention phrase boshay. Behavior expected moto.

---

## Ki baki ache (model lagbe)

Ei module gulo CPU tei verify hoyeche. Jegulo GPU/model lage:
- RQ1 uncertainty calibration (semantic entropy vs self-consistency vs verbalized) - **Kaggle T4**
- RAG correction quality (FAISS + BGE-M3) - retrieval index lage
- Full end-to-end number on BenHalluEval 12K
- Baselines (Raw SLM, CoT, Uniform RAG, CoVe) - Kaggle

Ei gulo `notebooks/SETU_Kaggle_Full.ipynb` te ready - T4 x2 te run korlei result asbe.
