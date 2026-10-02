"""Pooling A/B: mean-pooling vs CLS-token for Arctic Embed 2.0 L.

Empirical test — the model card mandates CLS pooling; the code uses mean.
This measures retrieval impact on memory-style fact/query pairs.

Run:  python3 pooling_ab.py   (from this directory)
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from limbic.embeddings import OnnxEmbedder  # noqa: E402

import numpy as np

# Memory-style corpus: facts as Limbic stores them, queries as users ask.
# Synthetic data shaped like the real workload: preferences, opaque IDs,
# hosted-service geography, locale, tech-stack, deployment topology.
FACTS = [
    "User prefers concise responses without tables in chat interfaces.",
    "The weekly model monitoring cron job id is c4e9a2f17b3d.",
    "GPG automation key fingerprint 4F2A8C15D90B7E36F58C21A4B09D5E7731C2F8A0.",
    "Camoufox is a Firefox fork with anti-fingerprinting patches for stealth browsing.",
    "The rendering service is deployed on Fly.io in Frankfurt.",
    "User is vegetarian and prefers meat-free alternatives by default.",
    "SQLite vec0 virtual tables store embeddings with cosine distance metric.",
    "Trust decay uses an Ebbinghaus curve with an adaptive half-life around 29 days.",
    "The municipal library catalogue system requires login before search.",
    "Entity extraction combines spaCy NER with rule-based known patterns.",
    "User's timezone is Europe/Berlin (CET, UTC+1).",
    "The deployment runs five service instances on a single Debian host.",
]

# Query → the single correct fact index. Paraphrase + lexical + hard-negative
# cases; several queries could semi-match multiple facts (deliberate).
QUERIES = [
    ("how does the user want me to format replies", 0),
    ("what cron job watches models", 1),
    ("what is the automation key fingerprint", 2),
    ("which browser resists fingerprinting", 3),
    ("where is the rendering service hosted", 4),
    ("dietary preferences", 5),
    ("how are vectors stored in the database", 6),
    ("how do memories fade over time", 7),
    ("library account login order", 8),
    ("how are names detected in facts", 9),
    ("what timezone should I use for scheduling", 10),
    ("how many service instances run on the server", 11),
    # paraphrase stress
    ("fingerprint of the gpg key", 2),
    ("stealth browser engine choice", 3),
    ("vegetarian", 5),
    ("forgetting curve for trust scores", 7),
]


def embed_cls(session, tokenizer, text, prefix):
    """CLS pooling: first token of last hidden state (model-card spec)."""
    encoding = tokenizer.encode(f"{prefix}{text}" if prefix else text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    emb = outputs[0][0, 0]  # CLS token
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def cosine(a, b):
    return float(np.dot(a, b))


def evaluate(name, vectors_facts, vectors_queries):
    hits, ranks, scores = [], [], []
    for i, (q, expected) in enumerate(QUERIES):
        sims = [cosine(vectors_queries[i], vf) for vf in vectors_facts]
        order = sorted(range(len(sims)), key=lambda j: -sims[j])
        rank = order.index(expected) + 1
        ranks.append(rank)
        scores.append(sims[expected])
        if rank == 1:
            hits.append(1)
        else:
            hits.append(0)
    hit1 = sum(hits) / len(QUERIES)
    hit3 = sum(1 for r in ranks if r <= 3) / len(QUERIES)
    mrr = sum(1.0 / r for r in ranks) / len(QUERIES)
    print(f"{name}: hit@1={hit1:.3f} hit@3={hit3:.3f} MRR={mrr:.3f} "
          f"avg-target-sim={sum(scores)/len(scores):.4f}")
    return hit1, hit3, mrr, ranks


def main():
    emb = OnnxEmbedder()
    emb._ensure_model()
    session, tokenizer = emb._session, emb._tokenizer
    prefix_f, prefix_q = "", "query: "

    print("loading mean-pool (current code)...")
    # Current code path — mean pooling via embed()
    facts_mean = [emb.embed(f) for f in FACTS]
    queries_mean = [emb.embed(q, is_query=True) for q, _ in QUERIES]

    print("computing CLS-pool (model-card spec)...")
    facts_cls = [embed_cls(session, tokenizer, f, prefix_f) for f in FACTS]
    queries_cls = [embed_cls(session, tokenizer, q, prefix_q) for q, _ in QUERIES]

    print(f"\ncorpus={len(FACTS)} facts, {len(QUERIES)} queries\n")
    m = evaluate("MEAN (current)", facts_mean, queries_mean)
    c = evaluate("CLS  (model card)", facts_cls, queries_cls)

    print("\nper-query ranks (target rank; 1 = top):")
    print("query                          MEAN  CLS")
    for i, (q, e) in enumerate(QUERIES):
        flag = "  <-- CLS wins" if c[3][i] < m[3][i] else ("  <-- MEAN wins" if m[3][i] < c[3][i] else "")
        print(f"  {q[:28]:30s} {m[3][i]:3d}   {c[3][i]:3d}{flag}")


if __name__ == "__main__":
    main()