# Kaggle e SETU Run Korar Guide - Step by Step

## Step 1: Kaggle Account
- kaggle.com e jao, Google diye login koro
- Phone verify koro (free GPU er jonno lage)

## Step 2: New Notebook Banao
- Kaggle -> Create -> New Notebook
- Title dao: SETU_Bengali_Hallucination
- Right side e:
  - Accelerator: GPU T4 x2
  - Internet: ON (important, model download er jonno)

## Step 3: Code Copy-Paste Koro - IMPORTANT! Branch Clone
Nicher full code ta first cell e paste koro, Shift+Enter chap:

**Merge na korleo data pabe, kintu branch specify korte hobe!**

```python
# SETU - Real Model Run on Kaggle - WITH BRANCH (merge chara)
!pip install -q transformers accelerate bitsandbytes sentence-transformers faiss-cpu scikit-learn

# IMPORTANT: -b flag diye amader arena branch clone korte hobe, main e code nai
!git clone -b arena/01a0c59d-a-selective-triage-and-correct https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98.git
%cd A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!ls -lh src/
# Dekhbe src/ folder e 12 ta file ache - mane branch thik clone hoyeche

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
- Run button chap, 5-6 min wait koro (first time model download 3GB)
- Output e dekhbe Bengali answer

## Step 5: Full SETU Pipeline (Next Cell)
```python
# Clone repo
!git clone https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98.git
%cd A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!pip install -r requirements.txt -q

# Run real pipeline
!python src/main.py --input "বাংলাদেশের রাজধানী কোথায় এবং এর জনসংখ্যা কত?"
```

## Troubleshooting
- CUDA out of memory -> model_name change to "Qwen/Qwen2.5-0.5B-Instruct" (choto model)
- Internet off thakle model download hobe na -> Settings e Internet ON koro
- Session expire -> Save kore rakho, abar run koro

## Next
- Result screenshot niye supervisor ke dekhao
- Amake bolo result ki aslo
