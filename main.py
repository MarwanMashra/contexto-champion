"""
contexto_terminal_v8.py - Contexto clone with GloVe wiki gigaword 300, per‑game Gaussian noise,
and top‑30 reveal on quit.

Run:
    python contexto_terminal_v8.py
    python contexto_terminal_v8.py --episodes 10 --noise_std 0.005
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import gensim.downloader as _api
    from gensim.models import KeyedVectors
except ImportError:
    print("gensim is required.  pip install gensim", file=sys.stderr)
    sys.exit(1)

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("nltk is required.  pip install nltk", file=sys.stderr)
    sys.exit(1)

# Ensure WordNet corpora
for corpus in ("wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus, quiet=True)

_LEMMA = WordNetLemmatizer()


def _lemmatise(word: str) -> str:
    for pos in ("n", "v", "a", "r", "s"):
        lw = _LEMMA.lemmatize(word, pos)
        if lw != word:
            return lw
    return word


# ------------------- Caching & loading ------------------------------------
CACHE_DIR = Path.home() / ".cache" / "contexto"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "gigaword_lemmatized.kv"


def _subset_kv(kv: KeyedVectors, tokens: List[str]) -> KeyedVectors:
    if hasattr(kv.__class__, "load_subset"):
        return kv.__class__.load_subset(kv, tokens)
    sub = KeyedVectors(vector_size=kv.vector_size)
    sub.add_vectors(tokens, np.asarray([kv[t] for t in tokens], dtype=np.float32))
    return sub


def _collapse_to_lemmas(kv: KeyedVectors) -> Tuple[KeyedVectors, Dict[str, str]]:
    lemma_to_token: Dict[str, str] = {}
    keep: List[str] = []
    for tok in kv.key_to_index:
        lem = _lemmatise(tok.lower())
        if lem not in lemma_to_token:
            lemma_to_token[lem] = tok
            keep.append(tok)
    print(f"Collapsing vocabulary: {len(kv)} -> {len(keep)} lemmas", file=sys.stderr)
    kv_small = _subset_kv(kv, keep)
    return kv_small, lemma_to_token


def get_kv_cached(skip_collapse: bool) -> Tuple[KeyedVectors, Dict[str, str], bool]:
    if CACHE_FILE.exists() and not skip_collapse:
        kv = KeyedVectors.load(str(CACHE_FILE), mmap="r")
        lem_map = {t.lower(): t for t in kv.key_to_index}
        return kv, lem_map, True

    kv_raw = _api.load("glove-wiki-gigaword-300")
    if skip_collapse:
        kv = kv_raw
        lem_map = {t.lower(): t for t in kv.key_to_index}
        return kv, lem_map, False

    kv, lem_map = _collapse_to_lemmas(kv_raw)
    kv.fill_norms()
    kv.save(str(CACHE_FILE))
    return kv, lem_map, False


# ------------------- Game mechanics --------------------------------------


def _pick_secret(lemmas: List[str]) -> str:
    today = _dt.date.today().isoformat()
    digest = hashlib.sha256(today.encode()).digest()
    return lemmas[int.from_bytes(digest[:8], "big") % len(lemmas)]


def _prepare(
    scores_base: np.ndarray, secret_idx: int, std: float, rng: np.random.Generator
):
    scores = scores_base.copy()
    if std > 0:
        noise = rng.normal(0.0, std, size=scores.shape)
        noise[secret_idx] = 0.0
        scores += noise
        if scores.max() >= scores[secret_idx]:
            scores[secret_idx] = scores.max() + 1e-6
    order = np.argsort(-scores)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(1, len(order) + 1)
    return scores, rank_of, order


def _show_top(kv: KeyedVectors, scores: np.ndarray, order: np.ndarray, n: int = 30):
    print("\nTop 30 words:")
    for i in range(n):
        idx = order[i]
        print(f"{i + 1:2d}. {kv.index_to_key[idx]:<15}  sim={scores[idx]:.3f}")


def _run_game(kv, lem_map, norms, secret_tok, std, rng):
    secret_idx = kv.key_to_index[secret_tok]
    base_scores = norms @ norms[secret_idx]
    scores, rank_of, order = _prepare(base_scores, secret_idx, std, rng)

    print(f"\nNew game (noise σ={std}). Type guesses; 'quit' to reveal & exit.\n")
    attempts = 0
    while True:
        try:
            raw = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession aborted.")
            sys.exit(0)

        if raw.lower() in ("", "quit", "exit"):
            print(f"Game over. The secret word was '{secret_tok.lower()}'.")
            _show_top(kv, scores, order)
            break

        lem = _lemmatise(raw.lower())
        tok = lem_map.get(lem)
        if tok is None:
            print("Word not in vocabulary, try again.")
            continue

        idx = kv.key_to_index[tok]
        attempts += 1
        rank = int(rank_of[idx])
        sim = float(scores[idx])

        if rank == 1:
            print(f"🎉 Correct in {attempts} guesses!\n")
            break
        print(f"Rank: {rank:,}   Similarity: {sim:.3f}")


# ------------------- CLI --------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Contexto CLI with quit‑to‑reveal feature.")
    p.add_argument("--secret", help="Force secret word.")
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--skip_collapse", action="store_true")
    args = p.parse_args()

    kv, lem_map, cached = get_kv_cached(args.skip_collapse)
    if cached:
        print("Loaded lemma‑collapsed vectors from cache.", file=sys.stderr)
    norms = kv.get_normed_vectors()
    rng = np.random.default_rng()

    for ep in range(1, args.episodes + 1):
        if args.secret and ep == 1:
            lem = _lemmatise(args.secret.lower())
            secret_tok = lem_map.get(lem)
            if secret_tok is None:
                print(f"Secret '{args.secret}' not in vocabulary.", file=sys.stderr)
                sys.exit(1)
        else:
            secret_tok = _pick_secret(list(kv.key_to_index))
        print(f"\n=== Episode {ep}/{args.episodes} ===")
        _run_game(kv, lem_map, norms, secret_tok, args.noise_std, rng)


if __name__ == "__main__":
    main()
