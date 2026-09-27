"""Dense retrieval with Qwen3-Embedding-0.6B + FAISS GPU.

Uses Qwen/Qwen3-Embedding-0.6B (614M, Apache-2.0, 100+ languages):
  - True Matryoshka (MRL) — 512-dim works correctly
  - MTEB multilingual retrieval: 64.64
  - sentence-transformers compatible via SentenceTransformer()
  - Requires prompt_name="query" for S1 entities, "document" for pool

VRAM budget on A10G 24GB at 512-dim:
  10M pool vectors × 512-dim × fp16 = 10.2 GB (FAISS index)
  Qwen3-Embedding model weights        ≈  1.2 GB
  Cross-encoder model (mDeBERTa)       ≈  0.6 GB
  Total                                ≈ 12.0 GB  — CE and index coexist ✓

Validated 2026-09-26: BGE-m3 is NOT Matryoshka; Qwen3-Embedding-0.6B
is the superior drop-in replacement (true MRL, better MTEB, same license tier).
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np


class DenseRetrieval:
    """Encode entities and retrieve dense nearest neighbors."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 device: str = "cuda", batch_size: int = 512,
                 truncate_dim: int = 512):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.bfloat16})
        self.device = device
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self._pool_eids: list[str] = []
        self._index = None
        self._gpu_res = None

    def _texts(self, records: dict) -> tuple[list[str], list[str]]:
        eids = list(records.keys())
        texts = []
        for eid in eids:
            r = records[eid]
            texts.append(f"{r[0]} {r[1]} {r[2]}".strip())
        return eids, texts

    def _encode(self, texts: list[str], tag: str = "",
                prompt_name: str | None = None) -> np.ndarray:
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device, model_kwargs={"torch_dtype": torch.bfloat16})
        t0 = time.time()
        kwargs = dict(batch_size=self.batch_size, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True)
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        emb = self.model.encode(texts, **kwargs).astype(np.float32)
        # truncate_dim=0 means no truncation (BGE-m3 is not Matryoshka).
        # Only set truncate_dim for models that explicitly support it
        # (nomic-embed-text-v1.5, mxbai-embed-large-v1).
        if self.truncate_dim and self.truncate_dim > 0 and emb.shape[1] > self.truncate_dim:
            emb = emb[:, :self.truncate_dim]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
        rate = len(texts) / max(time.time() - t0, 0.01)
        print(f"  [{tag}] {len(texts):,} → {emb.shape} "
              f"({rate:.0f}/s, {time.time()-t0:.0f}s)", flush=True)
        return emb

    # ── pool index ────────────────────────────────────────────────────

    def encode_and_index(self, pool: dict, use_gpu: bool = True,
                         fp16: bool = True, chunk: int = 250_000,
                         n_gpu: int = -1):
        """Encode pool records in chunks and build FAISS index.

        n_gpu: number of GPUs to use for FAISS index (-1 = all available).
        On g5.12xlarge (4× A10G), set n_gpu=4 for 4× faster search.
        Encoding itself is parallelised by splitting pool across GPU processes
        — call encode_and_index_multigpu() for that.
        """
        import faiss

        eids, texts = self._texts(pool)
        self._pool_eids = eids

        dim = self.truncate_dim or self.model.get_sentence_embedding_dimension()
        n_gpus_available = faiss.get_num_gpus() if use_gpu else 0
        n_gpus_use = n_gpus_available if n_gpu == -1 else min(n_gpu, n_gpus_available)

        if n_gpus_use > 1:
            # Multi-GPU flat index across all GPUs
            cpu_index = faiss.IndexFlatIP(dim)
            self._index = faiss.index_cpu_to_all_gpus(cpu_index)
            print(f"FAISS: using {n_gpus_use} GPUs (index_cpu_to_all_gpus)", flush=True)
        elif n_gpus_use == 1:
            self._gpu_res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = fp16
            self._index = faiss.GpuIndexFlatIP(self._gpu_res, dim, cfg)
        else:
            self._index = faiss.IndexFlatIP(dim)

        t0 = time.time()
        for start in range(0, len(texts), chunk):
            end = min(start + chunk, len(texts))
            emb = self._encode(texts[start:end], f"pool {start:,}–{end:,}",
                               prompt_name="document")
            self._index.add(emb)
            del emb
            gc.collect()

        elapsed = time.time() - t0
        print(f"FAISS index: {self._index.ntotal:,} vectors, dim={dim}, "
              f"gpus={n_gpus_use}, fp16={fp16}, {elapsed:.0f}s total", flush=True)

    def search(self, query_records: dict, top_k: int = 200
               ) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
        """Return (candidates, cosine_scores) per S1 entity.

        cosine_scores[sid][pool_eid] = cosine similarity (from FAISS search).
        These become feature #18 (embedding_cosine) in the pipeline.
        """
        eids, texts = self._texts(query_records)
        # Query entities = "query" prompt for Qwen3-Embedding
        q_emb = self._encode(texts, "query", prompt_name="query")

        t0 = time.time()
        scores, indices = self._index.search(q_emb, top_k)
        print(f"  FAISS search {len(eids):,}×{top_k}: {time.time()-t0:.1f}s", flush=True)
        del q_emb
        gc.collect()

        pool_arr = np.array(self._pool_eids)
        candidates: dict[str, list[str]] = {}
        cosines: dict[str, dict[str, float]] = {}
        for i, sid in enumerate(eids):
            mask = indices[i] >= 0
            cids = pool_arr[indices[i][mask]].tolist()
            sims = scores[i][mask].tolist()
            candidates[sid] = cids
            cosines[sid] = dict(zip(cids, sims))
        del scores, indices
        gc.collect()
        return candidates, cosines

    # ── search in chunks (for large S1 test sets) ────────────────────

    def search_chunked(self, query_records: dict, top_k: int = 200,
                       chunk: int = 50_000
                       ) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
        """Like search() but processes queries in chunks to limit RAM."""
        all_cands: dict[str, list[str]] = {}
        all_cos: dict[str, dict[str, float]] = {}

        eids = list(query_records.keys())
        for start in range(0, len(eids), chunk):
            batch_eids = eids[start:start + chunk]
            batch_records = {e: query_records[e] for e in batch_eids}
            c, s = self.search(batch_records, top_k)
            all_cands.update(c)
            all_cos.update(s)
            del c, s
            gc.collect()
            print(f"  Dense search progress: {min(start+chunk, len(eids)):,}/{len(eids):,}",
                  flush=True)
        return all_cands, all_cos

    # ── checkpoint ────────────────────────────────────────────────────

    def save_pool(self, path: Path):
        import faiss
        idx = self._index
        if hasattr(idx, "getDevice"):
            idx = faiss.index_gpu_to_cpu(idx)
        faiss.write_index(idx, str(path.with_suffix(".faiss")))
        np.save(path.with_suffix(".eids.npy"),
                np.array(self._pool_eids, dtype=object))
        print(f"Saved pool index → {path.stem}.*", flush=True)

    def load_pool(self, path: Path, use_gpu: bool = False, n_gpu: int = 0):
        import faiss
        self._index = faiss.read_index(str(path.with_suffix(".faiss")))
        self._pool_eids = np.load(
            path.with_suffix(".eids.npy"), allow_pickle=True).tolist()
        print(f"Loaded pool CPU index: {self._index.ntotal:,} vectors", flush=True)

    # ── multi-GPU parallel pool encoding ─────────────────────────────

    @staticmethod
    def encode_shard(args):
        """Worker: encode one shard of pool texts on a specific GPU.
        Called via multiprocessing for parallel encoding across GPUs.
        """
        model_name, truncate_dim, batch_size, texts, gpu_id, out_path = args
        import torch
        torch.cuda.set_device(gpu_id)
        from sentence_transformers import SentenceTransformer
        device_str = f"cuda:{gpu_id}"
        model = SentenceTransformer(model_name, device=device_str, model_kwargs={"torch_dtype": torch.bfloat16})
        model.max_seq_length = 128
        kwargs = dict(batch_size=256, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True,
                      prompt_name="document")
        emb = model.encode(texts, **kwargs).astype("float32")
        if truncate_dim and emb.shape[1] > truncate_dim:
            emb = emb[:, :truncate_dim]
            import numpy as np
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
        np.save(out_path, emb)
        print(f"  GPU {gpu_id}: encoded {len(texts):,} → saved {out_path}",
              flush=True)

    def encode_and_index_multigpu(self, pool: dict, tmp_dir: Path,
                                   fp16: bool = True):
        """Encode pool using ALL available GPUs in parallel, then merge.

        Splits pool into N shards (one per GPU), encodes in parallel via
        multiprocessing, then concatenates and builds FAISS index.
        Gives ~N× speedup on g5.12xlarge (4× A10G).
        """
        import faiss
        import multiprocessing as mp

        n_gpus = faiss.get_num_gpus()
        if n_gpus <= 1:
            print("Single GPU — using standard encode_and_index", flush=True)
            return self.encode_and_index(pool, use_gpu=True, fp16=fp16)

        print(f"Multi-GPU pool encoding: {n_gpus} GPUs", flush=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        eids, texts = self._texts(pool)
        self._pool_eids = eids

        # Check and combine existing part files into full shard files
        for i in range(n_gpus):
            shard_p = tmp_dir / f"shard_{i}.npy"
            p0 = tmp_dir / f"shard_{i}_part0.npy"
            p1 = tmp_dir / f"shard_{i}_part1.npy"
            if not shard_p.exists() and p0.exists() and p1.exists():
                print(f"Combining existing parts for shard_{i}...", flush=True)
                emb0 = np.load(str(p0))
                emb1 = np.load(str(p1))
                full_emb = np.concatenate([emb0, emb1], axis=0)
                np.save(str(shard_p), full_emb)
                del emb0, emb1, full_emb
                gc.collect()
                p0.unlink()
                p1.unlink()
                print(f"  Combined existing parts → shard_{i}.npy", flush=True)

        # Check existing full shards on disk
        shard_paths = [tmp_dir / f"shard_{i}.npy" for i in range(n_gpus)]
        missing_gpus = [i for i, p in enumerate(shard_paths) if not p.exists()]

        if missing_gpus:
            print(f"Missing shards to encode: {missing_gpus} — checking existing part files...", flush=True)
            shard_size = (len(texts) + n_gpus - 1) // n_gpus
            tasks = []
            for shard_id in missing_gpus:
                start = shard_id * shard_size
                end = min(start + shard_size, len(texts))
                shard_texts = texts[start:end]
                mid = len(shard_texts) // 2
                
                p0 = tmp_dir / f"shard_{shard_id}_part0.npy"
                p1 = tmp_dir / f"shard_{shard_id}_part1.npy"
                
                if not p0.exists():
                    sub_texts = shard_texts[:mid]
                    sub_chunk = (len(sub_texts) + n_gpus - 1) // n_gpus
                    for g in range(n_gpus):
                        s_idx = g * sub_chunk
                        e_idx = min(s_idx + sub_chunk, len(sub_texts))
                        t_slice = sub_texts[s_idx:e_idx]
                        if t_slice:
                            out_path = str(tmp_dir / f"shard_{shard_id}_part0_sub{g}.npy")
                            tasks.append((g, t_slice, out_path))
                else:
                    print(f"  Part file {p0.name} already exists on disk — skipping!", flush=True)

                if not p1.exists():
                    sub_texts = shard_texts[mid:]
                    sub_chunk = (len(sub_texts) + n_gpus - 1) // n_gpus
                    for g in range(n_gpus):
                        s_idx = g * sub_chunk
                        e_idx = min(s_idx + sub_chunk, len(sub_texts))
                        t_slice = sub_texts[s_idx:e_idx]
                        if t_slice:
                            out_path = str(tmp_dir / f"shard_{shard_id}_part1_sub{g}.npy")
                            tasks.append((g, t_slice, out_path))
                else:
                    print(f"  Part file {p1.name} already exists on disk — skipping!", flush=True)

            if tasks:
                args_list = []
                for gpu_id, sub_texts, out_path in tasks:
                    args_list.append((
                        self.model_name, self.truncate_dim, 512,
                        sub_texts, gpu_id, out_path,
                    ))

                t0 = time.time()
                if hasattr(self, "model") and self.model is not None:
                    del self.model
                    self.model = None
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=len(args_list)) as pool_mp:
                    pool_mp.map(DenseRetrieval.encode_shard, args_list)
                print(f"Remaining sub-shards encoded across {n_gpus} GPUs in {time.time()-t0:.0f}s", flush=True)

            # Combine sub-parts into part0/part1 and full shard files
            for shard_id in missing_gpus:
                for part_id in [0, 1]:
                    part_path = tmp_dir / f"shard_{shard_id}_part{part_id}.npy"
                    if not part_path.exists():
                        sub_files = [tmp_dir / f"shard_{shard_id}_part{part_id}_sub{g}.npy" for g in range(n_gpus)]
                        if all(f.exists() for f in sub_files):
                            print(f"Combining {len(sub_files)} sub-parts for shard_{shard_id}_part{part_id}...", flush=True)
                            arrs = [np.load(str(f)) for f in sub_files]
                            merged = np.concatenate(arrs, axis=0)
                            np.save(str(part_path), merged)
                            del arrs, merged
                            gc.collect()
                            for f in sub_files:
                                f.unlink()
                            print(f"  Saved combined {part_path.name}", flush=True)

                p0 = tmp_dir / f"shard_{shard_id}_part0.npy"
                p1 = tmp_dir / f"shard_{shard_id}_part1.npy"
                if p0.exists() and p1.exists():
                    emb0 = np.load(str(p0))
                    emb1 = np.load(str(p1))
                    full_emb = np.concatenate([emb0, emb1], axis=0)
                    np.save(str(tmp_dir / f"shard_{shard_id}.npy"), full_emb)
                    del emb0, emb1, full_emb
                    gc.collect()
                    p0.unlink()
                    p1.unlink()
                    print(f"  Combined parts → shard_{shard_id}.npy", flush=True)
        else:
            print("All shards already exist on disk! Skipping encoding.", flush=True)

        # Merge shards into FAISS CPU index (uses 186GB RAM + 48 vCPUs, zero GPU OOM)
        dim = self.truncate_dim or 512
        self._index = faiss.IndexFlatIP(dim)
        print("Building FAISS CPU index on 186GB RAM across 48 vCPUs...", flush=True)

        t0 = time.time()
        for gpu_id in range(n_gpus):
            shard_path = tmp_dir / f"shard_{gpu_id}.npy"
            emb = np.load(str(shard_path))
            self._index.add(emb)
            print(f"  Added shard_{gpu_id}.npy ({len(emb):,} vectors)", flush=True)
            del emb
            gc.collect()

        print(f"FAISS CPU index built: {self._index.ntotal:,} vectors, "
              f"dim={dim}, {time.time()-t0:.1f}s total", flush=True)
