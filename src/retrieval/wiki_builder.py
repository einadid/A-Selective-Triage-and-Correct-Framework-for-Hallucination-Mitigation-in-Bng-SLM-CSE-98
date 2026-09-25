"""
Real Bengali Wikipedia FAISS Index Builder - Week 2
Builds FAISS index from Bengali Wikipedia (bnwiki) for SETU RAG.

Supports 3 modes:
1. HF Wikipedia dataset: load_dataset("wikipedia", "20231101.bn") - easiest for Kaggle
2. Local dump: bnwiki-latest-pages-articles.xml.bz2 -> parse -> chunk
3. Fallback demo corpus (if no internet)

Uses BAAI/bge-m3 embeddings (multilingual, good for Bengali)
Hybrid with BM25.

Usage on Kaggle:
  from src.retrieval.wiki_builder import WikiIndexBuilder
  builder = WikiIndexBuilder(config)
  builder.build_from_hf(num_articles=5000)  # 5000 articles ~ 50k passages
  builder.save()

Then BengaliRetriever will auto-load it.
"""

import os
import pickle
import json
import re
from typing import List, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except:
    ST_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

from rank_bm25 import BM25Okapi
from tqdm import tqdm


class WikiIndexBuilder:
    def __init__(self, config=None, embedding_model: str = "BAAI/bge-m3"):
        self.config = config
        self.embedding_model_name = embedding_model
        if config and hasattr(config, 'retrieval'):
            self.embedding_model_name = config.retrieval.embedding_model

        self.chunk_size = 256
        self.chunk_overlap = 32
        if config and hasattr(config, 'retrieval'):
            self.chunk_size = config.retrieval.chunk_size
            self.chunk_overlap = config.retrieval.chunk_overlap

        self.index_path_bn = "data/index/bn_faiss.index"
        self.passage_path_bn = "data/index/bn_faiss_passages.pkl"
        self.index_path_en = "data/index/en_faiss.index"
        self.passage_path_en = "data/index/en_faiss_passages.pkl"
        if config and hasattr(config, 'retrieval'):
            self.index_path_bn = config.retrieval.index_path_bn
            self.passage_path_bn = self.index_path_bn.replace(".index", "_passages.pkl")
            self.index_path_en = config.retrieval.index_path_en
            self.passage_path_en = self.index_path_en.replace(".index", "_passages.pkl")

        self.embedder = None

    def _load_embedder(self):
        if self.embedder is None:
            if not ST_AVAILABLE:
                raise ImportError("sentence-transformers not installed")
            print(f"[WikiBuilder] Loading embedder {self.embedding_model_name}")
            try:
                self.embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(f"[WikiBuilder] Failed {self.embedding_model_name}: {e}, fallback to multilingual-e5")
                self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk Bengali text into passages"""
        # Clean
        text = text.strip()
        if len(text) < 50:
            return []

        # Split by sentences first (Bengali danda)
        sentences = re.split(r'[।\n]+', text)
        chunks = []
        current_chunk_words = []
        current_len = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()
            if current_len + len(words) > self.chunk_size:
                # Save current
                if current_len > 20:  # min length
                    chunks.append(" ".join(current_chunk_words))
                # Overlap
                overlap_words = current_chunk_words[-self.chunk_overlap:] if len(current_chunk_words) > self.chunk_overlap else []
                current_chunk_words = overlap_words + words
                current_len = len(current_chunk_words)
            else:
                current_chunk_words.extend(words)
                current_len += len(words)

        if current_len > 20:
            chunks.append(" ".join(current_chunk_words))

        return chunks

    def _collect_from_hf_wikipedia(self, lang: str = "bn", num_articles: int = 5000, min_length: int = 200) -> List[str]:
        """Load Wikipedia from HF datasets"""
        if not HF_AVAILABLE:
            raise ImportError("datasets library not available")

        print(f"[WikiBuilder] Loading Wikipedia HF dataset lang={lang}, num_articles={num_articles}")
        # Try different dataset names
        ds = None
        tried = []
        for dataset_name in ["wikimedia/wikipedia", "wikipedia"]:
            for date in ["20231101.bn", "20231101.en", "20231101"]:
                try:
                    # For wikimedia/wikipedia, use config like "20231101.bn"
                    # For wikipedia, use "20220301.bn" etc
                    if dataset_name == "wikimedia/wikipedia":
                        ds = load_dataset(dataset_name, f"{date}", split="train", streaming=False)
                    else:
                        # old wikipedia dataset
                        ds = load_dataset(dataset_name, f"20220301.{lang}", split="train")
                    print(f"[WikiBuilder] Loaded {dataset_name} {date}: {len(ds)} articles")
                    break
                except Exception as e:
                    tried.append(f"{dataset_name}/{date}: {e}")
                    continue
            if ds is not None:
                break

        if ds is None:
            print(f"[WikiBuilder] All HF attempts failed: {tried[:3]}")
            # Try alternative: load_dataset("wikipedia", "20231101.bn") without streaming
            # Or use bnwiki dump from HF: "rahular/wikipedia-bengali" ?
            try:
                ds = load_dataset("rahular/wikipedia-bengali", split="train")
                print(f"[WikiBuilder] Loaded rahular/wikipedia-bengali: {len(ds)}")
            except Exception as e:
                print(f"[WikiBuilder] Fallback also failed: {e}")
                raise RuntimeError(f"Could not load Wikipedia HF dataset: {tried}")

        passages = []
        # Sample num_articles
        # If streaming, take first num_articles
        count = 0
        for article in tqdm(ds, total=min(num_articles, len(ds) if hasattr(ds, '__len__') else num_articles)):
            if count >= num_articles:
                break
            text = article.get("text", "") or article.get("article", "") or article.get("content", "")
            title = article.get("title", "")
            if len(text) < min_length:
                continue
            # Add title as context
            full_text = f"{title}. {text}" if title else text
            chunks = self._chunk_text(full_text)
            passages.extend(chunks)
            count += 1
            if count % 500 == 0:
                print(f"[WikiBuilder] Processed {count} articles -> {len(passages)} passages")

        print(f"[WikiBuilder] Collected {len(passages)} passages from {count} articles")
        return passages

    def _collect_from_local_dump(self, dump_path: str, max_passages: int = 50000) -> List[str]:
        """Parse local Wikipedia XML dump (bnwiki-latest-pages-articles.xml.bz2)"""
        print(f"[WikiBuilder] Parsing local dump {dump_path}")
        # For simplicity, support txt folder or jsonl
        passages = []
        if os.path.isdir(dump_path):
            for fname in os.listdir(dump_path):
                fpath = os.path.join(dump_path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        chunks = self._chunk_text(text)
                        passages.extend(chunks)
                        if len(passages) >= max_passages:
                            break
                except:
                    continue
        else:
            # Try jsonl
            with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        chunks = self._chunk_text(text)
                        passages.extend(chunks)
                        if len(passages) >= max_passages:
                            break
                    except:
                        continue
        return passages[:max_passages]

    def _build_faiss_index(self, passages: List[str], save_path_index: str, save_path_passages: str, batch_size: int = 256):
        """Embed passages and build FAISS"""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed")

        self._load_embedder()

        print(f"[WikiBuilder] Embedding {len(passages)} passages with {self.embedding_model_name} (batch={batch_size})")
        # Batch embedding to avoid OOM
        all_embeddings = []
        for i in tqdm(range(0, len(passages), batch_size)):
            batch = passages[i:i+batch_size]
            embs = self.embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(embs)
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        print(f"[WikiBuilder] Embeddings shape: {embeddings.shape}")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        index.add(embeddings)
        print(f"[WikiBuilder] FAISS index built: {index.ntotal} vectors, dim={dim}")

        # Save
        os.makedirs(os.path.dirname(save_path_index), exist_ok=True)
        faiss.write_index(index, save_path_index)
        with open(save_path_passages, "wb") as f:
            pickle.dump(passages, f)
        print(f"[WikiBuilder] Saved index to {save_path_index} and passages to {save_path_passages}")

        # Also build BM25 for hybrid (will be loaded by BengaliRetriever)
        print(f"[WikiBuilder] Building BM25...")
        tokenized = [p.split() for p in passages]
        bm25 = BM25Okapi(tokenized)
        bm25_path = save_path_passages.replace("_passages.pkl", "_bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"[WikiBuilder] Saved BM25 to {bm25_path}")

        return index, passages

    # ---------------- Public API ----------------
    def build_from_hf(self, lang: str = "bn", num_articles: int = 5000, max_passages: int = 50000):
        """Build from HF Wikipedia"""
        try:
            passages = self._collect_from_hf_wikipedia(lang=lang, num_articles=num_articles)
        except Exception as e:
            print(f"[WikiBuilder] HF build failed: {e}, using demo corpus")
            passages = self._demo_corpus()

        # Truncate if too many
        if len(passages) > max_passages:
            passages = passages[:max_passages]

        save_idx = self.index_path_bn if lang == "bn" else self.index_path_en
        save_psg = self.passage_path_bn if lang == "bn" else self.passage_path_en

        return self._build_faiss_index(passages, save_idx, save_psg)

    def build_from_local(self, dump_path: str, lang: str = "bn", max_passages: int = 50000):
        passages = self._collect_from_local_dump(dump_path, max_passages=max_passages)
        if len(passages) == 0:
            print("[WikiBuilder] No passages from local dump, using demo")
            passages = self._demo_corpus()
        save_idx = self.index_path_bn if lang == "bn" else self.index_path_en
        save_psg = self.passage_path_bn if lang == "bn" else self.passage_path_en
        return self._build_faiss_index(passages, save_idx, save_psg)

    def build_demo(self):
        """Build small demo index for quick testing (no internet needed)"""
        passages = self._demo_corpus() * 20  # expand
        return self._build_faiss_index(passages, self.index_path_bn, self.passage_path_bn)

    def _demo_corpus(self) -> List[str]:
        """Demo Bengali corpus for offline testing"""
        return [
            "বাংলাদেশের রাজধানী ঢাকা। ঢাকা বাংলাদেশের সর্ববৃহৎ শহর এবং বাণিজ্যিক কেন্দ্র।",
            "২০২২ সালের জনশুমারি অনুযায়ী ঢাকা জেলার জনসংখ্যা প্রায় ১ কোটি ৪৫ লাখ। ঢাকা মহানগরীর জনসংখ্যা প্রায় ২ কোটি ২০ লাখ।",
            "ঢাকা শহর বুড়িগঙ্গা নদীর তীরে অবস্থিত। এটি ১৬০৮ সালে প্রতিষ্ঠিত হয়।",
            "বাংলাদেশের স্বাধীনতা দিবস ২৬ মার্চ এবং বিজয় দিবস ১৬ ডিসেম্বর। বাংলাদেশ ১৯৭১ সালে স্বাধীনতা লাভ করে।",
            "বাংলাদেশের আয়তন ১,৪৭,৫৭০ বর্গকিলোমিটার। এটি দক্ষিণ এশিয়ার একটি দেশ।",
            "বাংলাদেশের জাতীয় সংসদ ভবন ঢাকার শেরেবাংলা নগরে অবস্থিত। এটি লুই আই কান দ্বারা ডিজাইন করা।",
            "বাংলাদেশের জাতীয় ফুল শাপলা এবং জাতীয় পাখি দোয়েল। জাতীয় পশু রয়েল বেঙ্গল টাইগার।",
            "পদ্মা সেতুর দৈর্ঘ্য ৬.১৫ কিলোমিটার। এটি বাংলাদেশের দীর্ঘতম সেতু।",
            "রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার পান। তিনি বাংলা সাহিত্যের অন্যতম শ্রেষ্ঠ কবি।",
            "কাজী নজরুল ইসলাম বাংলাদেশের জাতীয় কবি। তিনি বিদ্রোহী কবি হিসেবে পরিচিত।",
            "সুন্দরবন বিশ্বের বৃহত্তম ম্যানগ্রোভ বন। এটি বাংলাদেশ ও ভারতে অবস্থিত।",
            "বাংলা ভাষা বাংলাদেশের রাষ্ট্রভাষা। ১৯৫২ সালের ২১ ফেব্রুয়ারি ভাষা আন্দোলন হয়।",
            "বাংলাদেশের প্রধান নদী পদ্মা, মেঘনা, যমুনা। এই তিন নদীর মিলনস্থল চাঁদপুর।",
            "ঢাকা বিশ্ববিদ্যালয় ১৯২১ সালে প্রতিষ্ঠিত হয়। এটি বাংলাদেশের প্রাচীনতম বিশ্ববিদ্যালয়।",
            "বাংলাদেশের মুদ্রার নাম টাকা। ১ ডলার প্রায় ১১৭ টাকা।",
        ]

    def load_existing(self, lang: str = "bn"):
        """Check if index exists"""
        path = self.index_path_bn if lang == "bn" else self.index_path_en
        return os.path.exists(path)

    def get_stats(self):
        """Get index stats"""
        stats = {}
        for lang, idx_path, psg_path in [("bn", self.index_path_bn, self.passage_path_bn), ("en", self.index_path_en, self.passage_path_en)]:
            if os.path.exists(idx_path):
                try:
                    index = faiss.read_index(idx_path)
                    with open(psg_path, "rb") as f:
                        passages = pickle.load(f)
                    stats[lang] = {"n_vectors": index.ntotal, "n_passages": len(passages), "path": idx_path}
                except Exception as e:
                    stats[lang] = {"error": str(e)}
            else:
                stats[lang] = {"exists": False}
        return stats


if __name__ == "__main__":
    from src.config import SETUConfig
    cfg = SETUConfig()
    builder = WikiIndexBuilder(cfg)
    # Build demo for testing
    builder.build_demo()
    print(builder.get_stats())
