# Kaggle e SETU Run Korar Guide - Step by Step

> **UPDATE: Repo ekhon PUBLIC + main branch e full code ache.**
> Tai token lagbe na, `-b` flag lagbe na - simple clone command e shob kaj korbe.

## Step 1: Kaggle Account
- kaggle.com e jao, Google diye login koro
- Phone verify koro (free GPU er jonno lage)

## Step 2: New Notebook Banao
- Kaggle -> Create -> New Notebook
- Title dao: SETU_Bengali_Hallucination
- Right side e:
  - Accelerator: GPU T4 x2
  - Internet: ON (important - model download + GitHub clone er jonno)

## Step 3: Code Copy-Paste Koro
Nicher full code ta first cell e paste koro, Shift+Enter chap:

```python
# SETU - Real Model Run on Kaggle
!pip install -q transformers accelerate bitsandbytes sentence-transformers faiss-cpu scikit-learn

# Repo public - token lagbe na, simple clone
!git clone https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98.git
%cd A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!ls -lh src/
# src/ folder e 12+ ta file dekha uchit (triage_router.py, setu_pipeline.py etc.)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("GPU:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Load Qwen2.5-1.5B 4-bit
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded!")

# Test Bengali
prompt = "বাংলাদেশের রাজধানী কোথায়?"
inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {prompt}\nA: {ans}")

# Test hallucination case
prompt2 = "ঢাকার জনসংখ্যা কত? (2022 census অনুযায়ী)"
inputs2 = tokenizer(f"User: {prompt2}\nAssistant:", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs2 = model.generate(**inputs2, max_new_tokens=150, temperature=0.7, do_sample=True)
    ans2 = tokenizer.decode(outputs2[0][inputs2.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {prompt2}\nA: {ans2}")
```

## Step 4: Run
- Run button chap, 5-6 min wait koro (first time model download ~3GB)
- Output e dekhbe Bengali answer
- Success hole `src/` folder er file list + GPU name + 2 ta Bengali answer dekhabe

## Step 5: Full SETU Pipeline (Next Cell)
Clone already hoye geche Step 3 e, tai abar clone korte hobe na - sudhu requirements install + run:

```python
%cd /kaggle/working/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!pip install -q -r requirements.txt

# Run real pipeline
!python src/main.py --input "বাংলাদেশের রাজধানী কোথায় এবং এর জনসংখ্যা কত?"
```

## Full Notebook
Ready-made 6-cell notebook o ache repo te: `notebooks/SETU_Kaggle_Full.ipynb`
Kaggle e: File -> Import Notebook -> ei file ta upload koro -> shob cell run koro.

## Latest Code Update Pete
Ami notun code push korle (main branch e), Kaggle e:
```python
%cd /kaggle/working/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!git pull origin main
```

## Troubleshooting
- **CUDA out of memory** -> model_name change to `Qwen/Qwen2.5-0.5B-Instruct` (choto model)
- **Internet off** -> `Could not resolve host: github.com` error ashe; Settings e Internet ON koro
- **Authentication failed / could not read Username** -> repo private hoye geche; GitHub Settings theke abar public koro
- **Session expire** -> Save kore rakho, abar run koro

## Next
- Result screenshot niye supervisor ke dekhao
- Amake bolo result ki aslo
