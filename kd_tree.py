"""
Project-1: Multi-dimensional Data Indexing and Similarity Query Processing
Scheme: k-d tree + LSH (MinHash)

Implements a two-phase search pipeline from scratch:
  - Phase 1 (k-d Tree): Indexes numeric and encoded-categorical attributes 
    (k <= 5) to efficiently resolve orthogonal range queries.
  - Phase 2 (LSH): Applies MinHash and Locality-Sensitive Hashing to textual 
    attributes of the filtered subset, identifying the Top-N most similar 
    items without relying on O(n^2) pairwise comparisons.

External libraries (e.g., scikit-learn, datasketch) are intentionally avoided 
to provide full transparency of the underlying algorithmic mechanics.
"""

from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from lsh import build_lsh_index

# ======================================================================
# Phase 1: k-d TREE (k <= 5 dimensions)
# ======================================================================

@dataclass
class KDNode:
    point: np.ndarray          # coordinates in the indexed k-dim space
    id_: int                   # row id / index in the original dataframe
    axis: int                  # splitting axis at this node
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    """
    k-d tree (median-split, alternating axis) supporting:
      - build(points, ids)
      - range_query(mins, maxs)   -> orthogonal range search
      - knn_query(point, n)       -> k nearest neighbours

    `points` must be a 2D numpy array of shape (N, k) with k <= 5,
    already numeric (categorical attributes must be label-encoded
    beforehand, see `encode_categorical`).
    """

    def __init__(self, dims: int):
        if not (1 <= dims <= 5):
            raise ValueError("Requires 1 <= k <= 5 dimensions")
        self.dims = dims
        self.root: Optional[KDNode] = None
        self.size = 0
  
    # ------------------------------- build --------------------------------
    def build(self, points: np.ndarray, ids: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        ids = np.asarray(ids)
        assert points.shape[1] == self.dims
        self.size = len(points)
        self.root = self._build(points, ids, depth=0)

    def _build(self, pts: np.ndarray, ids: np.ndarray, depth: int) -> Optional[KDNode]:
        n = len(pts)
        if n == 0:
            return None
        axis = depth % self.dims
        mid = n // 2

        # Uses argpartition to find the median in O(n) time instead of a full sort.
        # This optimizes the overall tree construction complexity to O(n log n).

        order = np.argpartition(pts[:, axis], mid)

        pts, ids = pts[order], ids[order]
        node = KDNode(point=pts[mid], id_=int(ids[mid]), axis=axis)
        node.left = self._build(pts[:mid], ids[:mid], depth + 1)
        node.right = self._build(pts[mid + 1:], ids[mid + 1:], depth + 1)
        return node

    # ------------------------------- range query --------------------------------
    def range_query(self, mins: Sequence[float], maxs: Sequence[float]) -> List[int]:
        """Orthogonal range search."""
        mins = np.asarray(mins, dtype=np.float64)
        maxs = np.asarray(maxs, dtype=np.float64)
        result: List[int] = []

        def report_subtree(node: Optional[KDNode]) -> None:
            """Bulk-reports all points in a subtree without spatial checks (O(P) time)."""
            if node is None:
                return
            result.append(node.id_)
            report_subtree(node.left)
            report_subtree(node.right)

        def recurse(node: Optional[KDNode], reg_min: np.ndarray, reg_max: np.ndarray) -> None:
            if node is None:
                return

            # If Reg(v) is contained in R, report all points in subtree(v)
            if np.all(reg_min >= mins) and np.all(reg_max <= maxs):
                report_subtree(node)
                return

            p = node.point
            
            # Check the point itself
            if np.all(p >= mins) and np.all(p <= maxs):
                result.append(node.id_)
                
            axis = node.axis
            
            # Prune and recurse appropriately, updating Reg(v) bounds
            if mins[axis] <= p[axis]:
                new_reg_max = reg_max.copy()
                new_reg_max[axis] = p[axis]
                recurse(node.left, reg_min, new_reg_max)
                
            if maxs[axis] >= p[axis]:
                new_reg_min = reg_min.copy()
                new_reg_min[axis] = p[axis]
                recurse(node.right, new_reg_min, reg_max)

        # The root's region Reg(root) is the entire space (-inf to +inf)
        initial_min = np.full(self.dims, -np.inf)
        initial_max = np.full(self.dims, np.inf)
        recurse(self.root, initial_min, initial_max)
        
        return result

    # ------------------------------- kNN --------------------------------
    def knn_query(self, point: Sequence[float], n: int = 5) -> List[Tuple[float, int]]:
        """Return the n nearest neighbours to `point` as (distance, id) pairs."""
        point = np.asarray(point, dtype=np.float64)
        heap: List[Tuple[float, int]] = []  # max-heap via negated distance

        def recurse(node: Optional[KDNode]) -> None:
            if node is None:
                return
            d = float(np.linalg.norm(node.point - point))
            if len(heap) < n:
                heapq.heappush(heap, (-d, node.id_))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, node.id_))
            axis = node.axis
            diff = point[axis] - node.point[axis]
            close, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            recurse(close)
            if len(heap) < n or abs(diff) < -heap[0][0]:
                recurse(far)

        recurse(self.root)
        return sorted([(-d, i) for d, i in heap])


def encode_categorical(series: pd.Series) -> Tuple[np.ndarray, Dict[Any, int]]:
    """Label-encode a categorical column to integers (needed to use it as a k-d axis)."""
    categories = {val: i for i, val in enumerate(sorted(series.astype(str).unique()))}
    encoded = series.astype(str).map(categories).to_numpy(dtype=np.float64)
    return encoded, categories


# ======================================================================
# End-to-end pipeline : k-d (phase 1) -> LSH (phase 2)
# ======================================================================

@dataclass
class KDAttribute:
    """Describes one of the (<=5) attributes used to build the k-d tree."""
    column: str
    kind: str  # "numeric" or "categorical"


def prepare_kd_matrix(df: pd.DataFrame, attributes: List[KDAttribute]) -> Tuple[np.ndarray, Dict[str, Dict[Any, int]]]:
    """Builds the (N, k) numeric matrix used to construct the k-d tree, encoding
    categorical columns to integers as needed."""
    cols = []
    encodings: Dict[str, Dict[Any, int]] = {}
    for attr in attributes:
        if attr.kind == "numeric":
            cols.append(pd.to_numeric(df[attr.column], errors="coerce").to_numpy(dtype=np.float64))
        elif attr.kind == "categorical":
            encoded, mapping = encode_categorical(df[attr.column])
            encodings[attr.column] = mapping
            cols.append(encoded)
        else:
            raise ValueError(f"Unknown kind {attr.kind!r} for column {attr.column}")
    matrix = np.vstack(cols).T
    return matrix, encodings


def run_kd_lsh_query(
    df: pd.DataFrame,
    kd_attributes: List[KDAttribute],
    ranges: Dict[str, Tuple[float, float]],
    text_col: str | List[str],
    top_n: int = 10,
    num_perm: int = 100,
    bands: int = 20,
    min_shingles: int = 2,
) -> Dict[str, Any]:
    """
    Executes the end-to-end k-d tree and LSH similarity pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset with a standard integer index.
    kd_attributes : List[KDAttribute]
        Attributes (up to 5) used to construct the k-d tree.
    ranges : Dict[str, Tuple[float, float]]
        Min/max filtering bounds for each k-d attribute.
    text_col : str or List[str]
        Textual column(s) used for LSH similarity (unioned if multiple).
    top_n : int, default=10
        Number of top similar pairs to retrieve.
    min_shingles : int, default=2
        Minimum tokens required per item to avoid trivial singleton collisions.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing execution timings, the filtered dataframe subset, 
        and the Top-N most similar pairs.
    """
    assert len(kd_attributes) <= 5, "k-d tree is restricted to k <= 5"

    # ---------- Phase 1 : build k-d tree & run the range query ----------
    t0 = time.perf_counter()
    matrix, encodings = prepare_kd_matrix(df, kd_attributes)
    valid_mask = ~np.isnan(matrix).any(axis=1)
    ids = df.index.to_numpy()[valid_mask]
    matrix = matrix[valid_mask]

    tree = KDTree(dims=len(kd_attributes))
    tree.build(matrix, ids)
    t_build = time.perf_counter() - t0

    mins = [ranges[a.column][0] for a in kd_attributes]
    maxs = [ranges[a.column][1] for a in kd_attributes]

    t0 = time.perf_counter()
    hit_ids = tree.range_query(mins, maxs)
    t_query = time.perf_counter() - t0

    subset = df.loc[hit_ids]

    # ---------- Phase 2 : LSH similarity search on the filtered subset ----------
    t0 = time.perf_counter()
    lsh, _, skipped_low_info = build_lsh_index(
        subset, text_col, num_perm=num_perm, bands=bands, min_shingles=min_shingles
    )
    t_lsh_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_pairs = lsh.top_n_pairs(n=top_n)
    t_lsh_query = time.perf_counter() - t0

    return {
        "kd_tree_size": tree.size,
        "matched_count": len(hit_ids),
        "subset": subset,
        "top_similar_pairs": top_pairs,   # list of (jaccard_est, id1, id2)
        "skipped_low_info_text": skipped_low_info,
        "categorical_encodings": encodings,
        "timings_sec": {
            "tree_build": t_build,
            "tree_query": t_query,
            "lsh_build": t_lsh_build,
            "lsh_query": t_lsh_query,
        },
    }
