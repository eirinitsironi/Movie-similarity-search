"""
Phase 2: Locality-Sensitive Hashing (LSH) for textual-attribute similarity.
"""
from __future__ import annotations
import ast
import hashlib
import heapq
import random
import re
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

_WORD_RE = re.compile(r"\w+")
_BIG_PRIME = (1 << 31) - 1  # 32-bit Mersenne prime (avoids int64 overflow in NumPy)


def to_shingle_set(value: Any) -> set:
    """
    Turn a cell value into a set of tokens ("shingles") to be hashed.
      - list/tuple/set values (e.g. genre_names, production_company_names)
        -> one shingle per element (lower-cased, trimmed)
      - plain free text (e.g. a review comment)
        -> word-level shingles (set of lower-cased words)
    """
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return {str(v).strip().lower() for v in parsed if str(v).strip()}
            except (ValueError, SyntaxError):
                pass
        return set(_WORD_RE.findall(s.lower()))
    return set()


class MinHasher:
    """Generates MinHash signatures using `num_perm` independent hash functions."""

    def __init__(self, num_perm: int = 100, seed: int = 42):
        self.num_perm = num_perm
        rng = random.Random(seed)
        self.a = np.array([rng.randint(1, _BIG_PRIME - 1) for _ in range(num_perm)], dtype=np.int64)
        self.b = np.array([rng.randint(0, _BIG_PRIME - 1) for _ in range(num_perm)], dtype=np.int64)

    @staticmethod
    def _hash_token(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % _BIG_PRIME

    def signature(self, tokens: Iterable[str]) -> np.ndarray:
        tokens = list(tokens)
        if not tokens:
            return np.full(self.num_perm, _BIG_PRIME, dtype=np.int64)
        h = np.array([self._hash_token(t) for t in tokens], dtype=np.int64)
        # (a*x + b) mod p, for every permutation, for every token -> take the min per permutation
        sig = ((self.a[:, None] * h[None, :] + self.b[:, None]) % _BIG_PRIME).min(axis=1)
        return sig


class LSHIndex:
    """
    Banded LSH index on top of MinHash signatures.
    num_perm must be divisible by `bands` (rows = num_perm / bands).
    Two items land in the same bucket (candidate pair) if at least one
    of their `bands` band-slices is identical -> avoids O(n^2) comparisons.
    """

    def __init__(self, num_perm: int = 100, bands: int = 20):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: List[Dict[int, List[int]]] = [dict() for _ in range(bands)]
        self.signatures: Dict[int, np.ndarray] = {}

    def insert(self, id_: int, signature: np.ndarray) -> None:
        self.signatures[id_] = signature
        for b in range(self.bands):
            band = signature[b * self.rows: (b + 1) * self.rows]
            key = hash(band.tobytes())
            self.buckets[b].setdefault(key, []).append(id_)

    def candidates(self, id_: int) -> set:
        signature = self.signatures[id_]
        cands: set = set()
        for b in range(self.bands):
            band = signature[b * self.rows: (b + 1) * self.rows]
            key = hash(band.tobytes())
            for cid in self.buckets[b].get(key, []):
                if cid != id_:
                    cands.add(cid)
        return cands

    def jaccard_estimate(self, id1: int, id2: int) -> float:
        s1, s2 = self.signatures[id1], self.signatures[id2]
        return float(np.mean(s1 == s2))

    def top_n_pairs(self, n: int = 10) -> List[Tuple[float, int, int]]:
        """
        Global Top-N similarity search. Uses a size-N min-heap to avoid storing 
        and sorting all candidate pairs, significantly improving memory efficiency 
        and performance.
        """
        seen: set = set()
        heap: List[Tuple[float, int, int]] = []  # min-heap of (score, id1, id2)
        for id_ in self.signatures:
            for c in self.candidates(id_):
                pair = (min(id_, c), max(id_, c))
                if pair in seen:
                    continue
                seen.add(pair)
                score = self.jaccard_estimate(*pair)
                entry = (score, pair[0], pair[1])
                if len(heap) < n:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        return sorted(heap, key=lambda x: -x[0])


def build_lsh_index(
    df: pd.DataFrame,
    text_col: str | List[str],
    num_perm: int = 100,
    bands: int = 20,
    min_shingles: int = 1,
) -> Tuple[LSHIndex, MinHasher, int]:
    """
    Builds the LSH index over the specified text column(s).

    Multiple columns can be provided to union their shingle sets per row.
    The `min_shingles` parameter skips rows with insufficient tokens (e.g., 
    single-tag entries) to prevent extreme hash collisions, which would 
    otherwise flood Top-N results and degrade search performance.

    Returns:
        Tuple of (lsh_index, hasher, n_skipped).
    """
    cols = [text_col] if isinstance(text_col, str) else list(text_col)
    hasher = MinHasher(num_perm=num_perm)
    lsh = LSHIndex(num_perm=num_perm, bands=bands)
    skipped = 0
    for idx, row in df[cols].iterrows():
        shingles: set = set()
        for c in cols:
            shingles |= to_shingle_set(row[c])
        if len(shingles) < min_shingles:
            skipped += 1
            continue
        sig = hasher.signature(shingles)
        lsh.insert(int(idx), sig)
    return lsh, hasher, skipped
