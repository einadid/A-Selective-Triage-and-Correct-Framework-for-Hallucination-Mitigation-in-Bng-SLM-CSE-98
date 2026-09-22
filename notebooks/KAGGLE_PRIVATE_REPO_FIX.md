# Kaggle Error Fix - Private Repo + Internet OFF

## Tomar Error:
```
Cloning into 'A-Selective-...'
fatal: unable to access 'https://github.com/...': Could not resolve host: github.com
```

## 2 ta problem:

### Problem 1: Kaggle e Internet OFF (Main Problem)
**Error "Could not resolve host: github.com" mane Internet OFF**

**Fix:**
1. Kaggle Notebook er right side e dekho **Settings** panel
2. **Internet** -> **ON** koro (toggle)
3. **Accelerator** -> **GPU T4 x2** select koro
4. Abar run koro

Internet ON na korle kono external site (github, huggingface) e jete parbe na.

### Problem 2: Repo Private (Second Problem)
Tomar repo private: `isPrivate: true`

Internet ON korar poro private repo clone korte token lagbe.

**3 ta Solution:**

#### Solution A: Repo Public Koro (Best - Recommended for Thesis)
Thesis er jonno public kora valo - portfolio hobe, supervisor dekhte parbe, job e dekhaite parbe.

**Kivabe public korbe:**
1. GitHub e jao: https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98/settings
2. Niche scroll koro -> **Danger Zone** -> **Change visibility** -> **Change to public**
3. Confirm koro

Public korle amar deya normal clone command e kaj korbe:
```bash
!git clone -b arena/01a0c59d-a-selective-triage-and-correct https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98.git
```

#### Solution B: Private rekhe Token diye Clone (Jodi public korte na chao)

**Step 1: GitHub Token banao**
1. GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)
2. Generate new token -> Classic
3. Expiration: 30 days
4. Scope: `repo` tick koro
5. Generate -> Token copy koro (ghp_xxxx...)

**Step 2: Kaggle e token diye clone**
```python
# Token ta Kaggle secret e rakho ba direct use koro (private notebook hole safe)
!git clone -b arena/01a0c59d-a-selective-triage-and-correct https://ghp_YOURTOKENHERE@github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98.git
```

**Security Note:** Token ta public notebook e dio na, private notebook e use koro. Kaggle e Secrets e add korte paro.

#### Solution C: Zip Upload Koro (Easiest, No Git Needed) - 100% Kaj Korbe

**Jodi git e jhamela lage, ei method e kono Internet token lagbe na:**

1. Tomar PC te GitHub repo theke zip download koro:
   - GitHub repo -> Code -> Download ZIP
   - Ba amar theke zip niye nao

2. Kaggle e:
   - New Notebook -> Right side -> **Add Input** -> **Upload** -> zip ta upload koro
   - Ba drag-drop koro

3. Notebook e:
```python
import zipfile
!ls /kaggle/input/
# Zip er naam dekhe unzip koro
!unzip /kaggle/input/your-zip-name/A-Selective-Triage-*.zip
!ls -lh
%cd A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!ls -lh src/
```

4. Tarpor normal:
```python
!pip install -q transformers accelerate bitsandbytes
!python run_demo.py
```

## Recommended Steps for You NOW:

1. **Kaggle e Internet ON koro** (right panel)
2. **Repo public koro** (2 min er kaj) - thesis er jonno best
3. Abar ei cell run koro:
```python
!git clone -b arena/01a0c59d-a-selective-triage-and-correct https://github.com/einadid/A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98.git
%cd A-Selective-Triage-and-Correct-Framework-for-Hallucination-Mitigation-in-Bng-SLM-CSE-98
!ls src/
```

4. Jodi public korte na chao, Solution B ba C follow koro

## Check:
- `!ls src/` e 12 ta file dekhale success
- Na dekhale screenshot dao

---
Need help? Amake bolo konta korbe - public, token, naki zip?
