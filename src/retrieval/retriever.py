"""
Retriever - Bengali Wikipedia FAISS + BM25 hybrid
"""
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json

class BengaliRetriever:
    def __init__(self, config, embedding_model=None):
        self.config = config
        self.embedding_model_name = config.retrieval.embedding_model if config else "BAAI/bge-m3"
        print(f"[Retriever] Loading embedding model {self.embedding_model_name}")
        try:
            self.embedder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"[Retriever] Failed to load {self.embedding_model_name}, fallback to multilingual-e5")
            self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")

        self.index_bn = None
        self.index_en = None
        self.passages_bn = []
        self.passages_en = []
        self.bm25_bn = None

        # Try load existing index
        self._try_load_index()

    def _try_load_index(self):
        bn_index_path = self.config.retrieval.index_path_bn if self.config else "data/index/bn_faiss.index"
        en_index_path = self.config.retrieval.index_path_en if self.config else "data/index/en_faiss.index"

        if os.path.exists(bn_index_path):
            try:
                self.index_bn = faiss.read_index(bn_index_path)
                with open(bn_index_path.replace(".index", "_passages.pkl"), "rb") as f:
                    self.passages_bn = pickle.load(f)
                print(f"[Retriever] Loaded BN index with {len(self.passages_bn)} passages")
                # Build BM25
                tokenized = [p.split() for p in self.passages_bn]
                self.bm25_bn = BM25Okapi(tokenized)
            except Exception as e:
                print(f"[Retriever] Failed to load BN index: {e}")

        if os.path.exists(en_index_path):
            try:
                self.index_en = faiss.read_index(en_index_path)
                with open(en_index_path.replace(".index", "_passages.pkl"), "rb") as f:
                    self.passages_en = pickle.load(f)
                print(f"[Retriever] Loaded EN index with {len(self.passages_en)} passages")
            except Exception as e:
                print(f"[Retriever] Failed to load EN index: {e}")

    def build_index_from_wiki_dump(self, wiki_dump_path: str, lang: str = "bn", max_passages: int = 100000):
        """
        Build FAISS index from Wikipedia dump
        wiki_dump_path: path to txt files or jsonl
        """
        print(f"[Retriever] Building index from {wiki_dump_path} for lang={lang}")
        passages = []

        # Support multiple formats
        if os.path.isdir(wiki_dump_path):
            for fname in os.listdir(wiki_dump_path)[:1000]:  # limit for demo
                fpath = os.path.join(wiki_dump_path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        # Chunk
                        chunk_size = self.config.retrieval.chunk_size if self.config else 256
                        overlap = self.config.retrieval.chunk_overlap if self.config else 32
                        words = text.split()
                        for i in range(0, len(words), chunk_size-overlap):
                            chunk = " ".join(words[i:i+chunk_size])
                            if len(chunk) > 50:
                                passages.append(chunk)
                                if len(passages) >= max_passages:
                                    break
                except:
                    continue
        else:
            # Assume jsonl
            with open(wiki_dump_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if len(text) > 50:
                            passages.append(text[:1000])
                            if len(passages) >= max_passages:
                                break
                    except:
                        continue

        print(f"[Retriever] Collected {len(passages)} passages, embedding...")
        embeddings = self.embedder.encode(passages, show_progress_bar=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine via normalized
        index.add(embeddings.astype(np.float32))

        # Save
        os.makedirs(os.path.dirname(self.config.retrieval.index_path_bn), exist_ok=True)
        if lang == "bn":
            faiss.write_index(index, self.config.retrieval.index_path_bn)
            with open(self.config.retrieval.index_path_bn.replace(".index", "_passages.pkl"), "wb") as f:
                pickle.dump(passages, f)
            self.index_bn = index
            self.passages_bn = passages
            tokenized = [p.split() for p in passages]
            self.bm25_bn = BM25Okapi(tokenized)
        else:
            faiss.write_index(index, self.config.retrieval.index_path_en)
            with open(self.config.retrieval.index_path_en.replace(".index", "_passages.pkl"), "wb") as f:
                pickle.dump(passages, f)
            self.index_en = index
            self.passages_en = passages

        print(f"[Retriever] Index built and saved for {lang}")

    def retrieve(self, query: str, top_k: int = 5, lang: str = "bn") -> List[Dict]:
        """Retrieve top_k passages for query"""
        if lang == "bn" and self.index_bn is None:
            print("[Retriever] BN index not loaded, returning empty")
            return []

        if lang == "en" and self.index_en is None:
            print("[Retriever] EN index not loaded, returning empty")
            return []

        index = self.index_bn if lang == "bn" else self.index_en
        passages = self.passages_bn if lang == "bn" else self.passages_en

        # Embed query
        # BGE-M3 needs prefix? For retrieval query, add instruction
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = index.search(q_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(passages):
                continue
            results.append({
                "passage": passages[idx],
                "score": float(score),
                "index": int(idx),
                "lang": lang
            })

        # Hybrid with BM25 for BN
        if lang == "bn" and self.bm25_bn is not None:
            tokenized_query = query.split()
            bm25_scores = self.bm25_bn.get_scores(tokenized_query)
            # Get top BM25
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
            for idx in top_bm25_idx:
                if bm25_scores[idx] > 0:
                    # Check if already in results
                    if not any(r["index"] == idx for r in results):
                        results.append({
                            "passage": passages[idx],
                            "score": float(bm25_scores[idx] / (np.max(bm25_scores)+1e-6)),  # normalize
                            "index": int(idx),
                            "lang": lang,
                            "source": "bm25"
                        })

        # Sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return results

    def retrieve_with_fallback(self, query: str, top_k: int = 5) -> Dict:
        """
        Adaptive cross-lingual retrieval: try BN first, if score < threshold -> EN fallback
        Returns dict with results and whether fallback was used
        """
        bn_results = self.retrieve(query, top_k=top_k, lang="bn")
        max_bn_score = max([r["score"] for r in bn_results]) if bn_results else 0
        threshold = self.config.retrieval.bn_threshold if self.config else 0.35

        if max_bn_score >= threshold or self.index_en is None:
            return {
                "results": bn_results,
                "fallback_used": False,
                "max_score": max_bn_score,
                "lang": "bn"
            }
        else:
            # Translate query to English for EN retrieval (simple: use translator or keep as is for demo)
            # In real thesis, use deep-translator or NLLB
            try:
                from deep_translator import GoogleTranslator
                en_query = GoogleTranslator(source='bn', target='en').translate(query)
            except:
                en_query = query  # fallback

            en_results = self.retrieve(en_query, top_k=top_k, lang="en")
            return {
                "results": en_results,
                "fallback_used": True,
                "max_score": max([r["score"] for r in en_results]) if en_results else 0,
                "lang": "en",
                "translated_query": en_query if 'en_query' in locals() else query,
                "bn_max_score": max_bn_score
            }
