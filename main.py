
"""
contexto_terminal.py - A minimal terminal clone of Contexto / Semantle.

How it works
------------
* Uses a pre‑trained GloVe 300‑d English embedding (Wikipedia + Gigaword, 400 k words).
* Picks a secret word deterministically from today's date (so everyone gets the same daily puzzle).
* Computes cosine‑similarity scores between the secret word vector and every other word vector, then
  ranks the vocabulary (rank 1 == secret word).
* During play every guess returns **rank** and **similarity** – just like contexto.me
  (smaller rank means you’re closer).

Limitations
-----------
* The real Contexto model is proprietary and bilingual, this script can’t reproduce its exact ranks.
* Because the whole 400 k‑word vocabulary is ranked in memory, first‑run load time is ~15 s on a
  laptop and ~1 GB RAM. Subsequent guesses are instant.
* English only (unless you swap in another KeyedVectors file).

Dependencies
------------
pip install "gensim>=4.3"

(Optional) If you want the top‑50 k common words instead of the full 400 k:
pip install wordfreq

Run
---
python contexto_terminal.py                # play today’s puzzle
python contexto_terminal.py --secret queen # custom secret word
python contexto_terminal.py --vocab_size 50000  # restrict vocab to 50k words

Enjoy!
"""

import argparse
import datetime as _dt
import hashlib
import sys

import numpy as np

try:
    import gensim.downloader as _api
except ImportError:
    print("gensim is required.  pip install gensim", file=sys.stderr)
    sys.exit(1)


def _pick_daily_secret(vocab):
    """Deterministically pick a secret word from today's date string."""
    today_str = _dt.date.today().isoformat()
    h = hashlib.sha256(today_str.encode()).digest()
    idx = int.from_bytes(h[:8], "big") % len(vocab)
    return vocab[idx]


def _load_vectors(model_name="glove-wiki-gigaword-300"):
    print(f"Loading embedding model '{model_name}' …", file=sys.stderr, flush=True)
    kv = _api.load(model_name)  # downloads on first run (~376 MB)
    print("Embedding loaded.", file=sys.stderr)
    return kv


def _restrict_vocab(kv, top_n):
    """Return a KeyedVectors limited to the `top_n` most common English words."""
    try:
        from wordfreq import top_n_list
    except ImportError:
        print("wordfreq not installed.  pip install wordfreq", file=sys.stderr)
        sys.exit(1)
    common = set(top_n_list("en", top_n))
    keep_keys = [w for w in kv.key_to_index if w in common]
    kv = kv.__class__.load_subset(kv, keep_keys)
    return kv


def _build_ranking(kv, secret):
    print("Pre‑computing similarity ranks …", file=sys.stderr)
    norms = kv.get_normed_vectors()
    secret_vec = norms[kv.key_to_index[secret]]
    scores = norms @ secret_vec
    ranking = np.argsort(-scores)  # descending
    rank_of = np.empty_like(ranking)
    rank_of[ranking] = np.arange(1, len(ranking) + 1)
    return scores, rank_of


def _interactive_loop(kv, secret, scores, rank_of):
    print("\n=== Contexto — terminal edition ===")
    print("Guess the secret word!  Type 'quit' or just press Enter to give up.\n")
    attempts = 0
    while True:
        guess = input("? ").strip().lower()
        if guess in ("", "quit", "exit"):
            print(f"Game over. The secret word was '{secret}'.")
            break
        if guess not in kv.key_to_index:
            print("Word not in vocabulary, try again.")
            continue
        idx = kv.key_to_index[guess]
        attempts += 1
        rank = int(rank_of[idx])
        similarity = float(scores[idx])
        if rank == 1:
            print(f"🎉 Correct in {attempts} guesses! The word is '{secret}'.")
            break
        print(f"Rank: {rank:,}   Similarity: {similarity:.3f}")


def main():
    p = argparse.ArgumentParser(description="Play a Contexto‑like game in the terminal.")
    p.add_argument("--secret", help="Set a custom secret word.")
    p.add_argument("--vocab_size", type=int, metavar="N",
                   help="Restrict to the N most common English words (requires wordfreq).")
    p.add_argument("--model", default="glove-wiki-gigaword-300",
                   help="gensim model name or path to a KeyedVectors file.")
    args = p.parse_args()

    kv = _load_vectors(args.model)
    if args.vocab_size:
        kv = _restrict_vocab(kv, args.vocab_size)

    secret = args.secret.lower() if args.secret else _pick_daily_secret(list(kv.key_to_index))
    if secret not in kv.key_to_index:
        print(f"Secret word '{secret}' is not in the vocabulary.", file=sys.stderr)
        sys.exit(1)

    scores, rank_of = _build_ranking(kv, secret)
    _interactive_loop(kv, secret, scores, rank_of)


if __name__ == "__main__":
    main()