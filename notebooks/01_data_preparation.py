"""
Notebook 01: Data Preparation - Kaggle ready
Run this on Kaggle to download datasets and build FAISS index
"""
# !pip install datasets faiss-cpu sentence-transformers rank-bm25

from datasets import load_dataset
import os, json, pickle
from tqdm import tqdm

# 1. TyDiQA Bengali
print("Loading TyDiQA...")
try:
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task")
    print(ds)
    # Filter Bengali
    # In secondary_task, each language is separate? Let's inspect
    print(ds["validation"][0])
except Exception as e:
    print(f"Error {e}, trying tydiqa plain")
    ds = load_dataset("tydiqa", "secondary_task")
    print(ds["validation"][0])

# 2. Build dummy FAISS for testing (real would use Wikipedia dump)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Building dummy FAISS index...")
model = SentenceTransformer("BAAI/bge-m3")
passages = [
    "ঢাকা বাংলাদেশের রাজধানী এবং বৃহত্তম শহর।",
    "বাংলাদেশ ১৯৭১ সালে স্বাধীনতা লাভ করে।",
    "বাংলা ভাষা বাংলাদেশের রাষ্ট্রভাষা।",
    "পদ্মা বাংলাদেশের প্রধান নদী।",
    "রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে নোবেল পুরস্কার পান।"
] * 20  # dummy expand

embeddings = model.encode(passages, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

os.makedirs("data/index", exist_ok=True)
faiss.write_index(index, "data/index/bn_faiss.index")
with open("data/index/bn_faiss_passages.pkl", "wb") as f:
    pickle.dump(passages, f)

print("Done! Index saved")
