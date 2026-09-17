"""
Project-1: Multi-dimensional Data Indexing and Similarity Query Processing
Scheme: Range Tree + LSH (MinHash)

Implements a highly optimized k-dimensional Range Tree combining:
1. Fractional Cascading for the final dimension (O(log N) Binary Search).
2. Bucket/Leaf Size limit (Vectorized Scanning) to prevent Python's recursion 
   depth and object-creation overhead from crashing on 1M+ datasets.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd

from lsh import build_lsh_index
from kd_tree import KDAttribute as TreeAttribute, prepare_kd_matrix

# ======================================================================
# Phase 1: OPTIMIZED RANGE TREE (k <= 5 dimensions)
# ======================================================================

class RangeNode:
    __slots__ = ['min_val', 'max_val', 'left', 'right', 'assoc', 'bucket_mat', 'bucket_ids']
    
    def __init__(self):
        self.min_val = 0.0
        self.max_val = 0.0
        self.left: Optional[RangeNode] = None
        self.right: Optional[RangeNode] = None
        self.assoc: Any = None
        
        # Γρήγοροι πίνακες NumPy για όταν σταματάμε το χτίσιμο (Bucket)
        self.bucket_mat: Optional[np.ndarray] = None
        self.bucket_ids: Optional[np.ndarray] = None


class RangeTree:
    def __init__(self, dims: int, leaf_size: int = 4096):
        self.dims = dims
        self.leaf_size = leaf_size  # Η "βαλβίδα ασφαλείας" για να μην κρασάρει η Python
        self.root: Optional[RangeNode] = None
        self.size = 0

    def build(self, matrix: np.ndarray, ids: np.ndarray) -> None:
        self.size = len(matrix)
        if self.size > 0:
            self.root = self._build(matrix, ids, 0)

    def _build(self, mat: np.ndarray, ids: np.ndarray, dim: int) -> Any:
        # 1. ΒΑΣΙΚΗ ΠΕΡΙΠΤΩΣΗ 1: Τελευταία Διάσταση (Fractional Cascading)
        if dim == self.dims - 1:
            order = np.argsort(mat[:, dim])
            return (mat[order, dim], ids[order])

        node = RangeNode()
        node.min_val = float(np.min(mat[:, dim]))
        node.max_val = float(np.max(mat[:, dim]))

        if len(mat) <= self.leaf_size:
            node.bucket_mat = mat
            node.bucket_ids = ids
            return node

        # 3. ΑΝΑΔΡΟΜΗ (Χτίσιμο δέντρου)
        # Χτίζουμε το δέντρο της επόμενης διάστασης
        node.assoc = self._build(mat, ids, dim + 1)

        # Χωρίζουμε στη μέση για την τρέχουσα διάσταση
        order = np.argsort(mat[:, dim])
        sorted_mat = mat[order]
        sorted_ids = ids[order]
        mid = len(sorted_mat) // 2

        if mid > 0:
            node.left = self._build(sorted_mat[:mid], sorted_ids[:mid], dim)
        if mid < len(sorted_mat):
            node.right = self._build(sorted_mat[mid:], sorted_ids[mid:], dim)

        return node

    def range_query(self, mins: Sequence[float], maxs: Sequence[float]) -> List[int]:
        mins_np = np.asarray(mins, dtype=np.float64)
        maxs_np = np.asarray(maxs, dtype=np.float64)
        result: List[int] = []
        
        if self.root is not None:
            self._query(self.root, mins_np, maxs_np, 0, result)
            
        return result

    def _query(self, node: Any, mins: np.ndarray, maxs: np.ndarray, dim: int, result: list) -> None:
        # --- Fractional Cascading (Τελευταία Διάσταση) ---
        if dim == self.dims - 1:
            vals, ids = node
            start_idx = np.searchsorted(vals, mins[dim], side='left')
            end_idx = np.searchsorted(vals, maxs[dim], side='right')
            if start_idx < end_idx:
                result.extend(ids[start_idx:end_idx].tolist())
            return

        # --- Κόψιμο (Pruning) αν βγήκαμε εκτός ορίων ---
        if maxs[dim] < node.min_val or mins[dim] > node.max_val:
            return

        # --- Αν πέσαμε σε Bucket, τα σκανάρουμε όλα αστραπιαία με NumPy ---
        if node.bucket_mat is not None:
            mat = node.bucket_mat
            mask = np.ones(len(mat), dtype=bool)
            for d in range(dim, self.dims):
                mask &= (mat[:, d] >= mins[d]) & (mat[:, d] <= maxs[d])
            result.extend(node.bucket_ids[mask].tolist())
            return

        # --- Πλήρης Επικάλυψη -> Πάμε στην επόμενη διάσταση (Assoc) ---
        if mins[dim] <= node.min_val and node.max_val <= maxs[dim]:
            self._query(node.assoc, mins, maxs, dim + 1, result)
            return

        # --- Μερική Επικάλυψη -> Ψάχνουμε τα παιδιά ---
        if node.left:
            self._query(node.left, mins, maxs, dim, result)
        if node.right:
            self._query(node.right, mins, maxs, dim, result)


# ======================================================================
# End-to-end pipeline : Range Tree (phase 1) -> LSH (phase 2)
# ======================================================================

def run_rangetree_lsh_query(
    df: pd.DataFrame,
    tree_attributes: List[TreeAttribute],
    ranges: Dict[str, Tuple[float, float]],
    text_col: str | List[str],
    top_n: int = 10,
    num_perm: int = 100,
    bands: int = 20,
    min_shingles: int = 2,
) -> Dict[str, Any]:

    assert len(tree_attributes) <= 5, "Range Tree is restricted to k <= 5"

    t0 = time.perf_counter()
    matrix, encodings = prepare_kd_matrix(df, tree_attributes)
    valid_mask = ~np.isnan(matrix).any(axis=1)
    ids = df.index.to_numpy()[valid_mask]
    matrix = matrix[valid_mask]

    # Leaf size = 4096 για να δουλεύει ΤΕΛΕΙΑ ακόμα και σε τεράστιο dataset
    tree = RangeTree(dims=len(tree_attributes), leaf_size=4096)
    tree.build(matrix, ids)
    t_build = time.perf_counter() - t0

    mins = [ranges[a.column][0] for a in tree_attributes]
    maxs = [ranges[a.column][1] for a in tree_attributes]

    t0 = time.perf_counter()
    hit_ids = tree.range_query(mins, maxs)
    t_query = time.perf_counter() - t0

    subset = df.loc[hit_ids]

    t0 = time.perf_counter()
    lsh, _, skipped_low_info = build_lsh_index(
        subset, text_col, num_perm=num_perm, bands=bands, min_shingles=min_shingles
    )
    t_lsh_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_pairs = lsh.top_n_pairs(n=top_n)
    t_lsh_query = time.perf_counter() - t0

    return {
        "tree_size": tree.size,
        "matched_count": len(hit_ids),
        "subset": subset,
        "top_similar_pairs": top_pairs,
        "skipped_low_info_text": skipped_low_info,
        "categorical_encodings": encodings,
        "timings_sec": {
            "tree_build": t_build,
            "tree_query": t_query,
            "lsh_build": t_lsh_build,
            "lsh_query": t_lsh_query,
        },
    }