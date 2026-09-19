"""
Project-1: Multi-dimensional Data Indexing and Similarity Query Processing
Scheme: Range Tree + LSH (MinHash)

Implements a highly optimized k-dimensional Range Tree combining:
1. Fractional Cascading for the final dimension (O(log N) Binary Search).
2. Bucket/Leaf Size limit (Vectorized Scanning) to prevent Python's recursion 
   depth and object-creation overhead from crashing on 1M+ datasets.
3. Fully dynamic capabilities (Insert, Delete, Update) that propagate 
   changes through both the main tree and all associated structures.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd

from lsh import build_lsh_index
from kd_tree import KDAttribute as TreeAttribute, prepare_kd_matrix

# ======================================================================
# Phase 1: OPTIMIZED DYNAMIC RANGE TREE (k <= 5 dimensions)
# ======================================================================

class RangeNode:
    __slots__ = ['min_val', 'max_val', 'left', 'right', 'assoc', 'bucket_mat', 'bucket_ids']
    
    def __init__(self):
        self.min_val = 0.0
        self.max_val = 0.0
        self.left: Optional[RangeNode] = None
        self.right: Optional[RangeNode] = None
        self.assoc: Any = None
        
        
        self.bucket_mat: Optional[np.ndarray] = None
        self.bucket_ids: Optional[np.ndarray] = None


class RangeTree:
    def __init__(self, dims: int, leaf_size: int = 4096):
        self.dims = dims
        self.leaf_size = leaf_size  
        self.root: Optional[RangeNode] = None
        self.size = 0

    def build(self, matrix: np.ndarray, ids: np.ndarray) -> None:
        """Mass Construction (Bottom-Up Bulk Loading)."""
        self.size = len(matrix)
        if self.size > 0:
            self.root = self._build(matrix, ids, 0)

    def _build(self, mat: np.ndarray, ids: np.ndarray, dim: int) -> Any:
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

   
        node.assoc = self._build(mat, ids, dim + 1)

        order = np.argsort(mat[:, dim])
        sorted_mat = mat[order]
        sorted_ids = ids[order]
        mid = len(sorted_mat) // 2

        if mid > 0:
            node.left = self._build(sorted_mat[:mid], sorted_ids[:mid], dim)
        if mid < len(sorted_mat):
            node.right = self._build(sorted_mat[mid:], sorted_ids[mid:], dim)

        return node

    # ------------------------------- insert --------------------------------
    def insert(self, coords: Sequence[float], id_: int) -> None:
        """Εισάγει δυναμικά ένα σημείο στο δέντρο και στις δομές Assoc."""
        pt = np.asarray(coords, dtype=np.float64)
        if self.size == 0:
            self.root = RangeNode()
            self.root.bucket_mat = np.array([pt])
            self.root.bucket_ids = np.array([id_])
            self.root.min_val = pt[0]
            self.root.max_val = pt[0]
        else:
            self.root = self._insert(self.root, pt, id_, 0)
        self.size += 1

    def _insert(self, node: Any, pt: np.ndarray, id_: int, dim: int) -> Any:
        # Base Case 1: Fractional Cascading
        if dim == self.dims - 1:
            vals, ids = node
            idx = np.searchsorted(vals, pt[dim])
            vals = np.insert(vals, idx, pt[dim])
            ids = np.insert(ids, idx, id_)
            return (vals, ids)

        # Base Case 2: Bucket
        if isinstance(node, RangeNode) and node.bucket_mat is not None:
            node.bucket_mat = np.vstack([node.bucket_mat, pt])
            node.bucket_ids = np.append(node.bucket_ids, id_)
            node.min_val = min(node.min_val, pt[dim])
            node.max_val = max(node.max_val, pt[dim])
            return node

        # Recursive Case
        node.min_val = min(node.min_val, pt[dim])
        node.max_val = max(node.max_val, pt[dim])

        #Routing 
        inserted_child = False
        if node.left and pt[dim] <= node.left.max_val:
            node.left = self._insert(node.left, pt, id_, dim)
            inserted_child = True
        elif node.right and pt[dim] >= node.right.min_val:
            node.right = self._insert(node.right, pt, id_, dim)
            inserted_child = True
        else:
            # Fallback: If it falls "between" or outside, insert it into the closest branch
            if node.left and node.right:
                if abs(pt[dim] - node.left.max_val) < abs(pt[dim] - node.right.min_val):
                    node.left = self._insert(node.left, pt, id_, dim)
                else:
                    node.right = self._insert(node.right, pt, id_, dim)
            elif node.left:
                node.left = self._insert(node.left, pt, id_, dim)
            elif node.right:
                node.right = self._insert(node.right, pt, id_, dim)
            inserted_child = True

  
        if inserted_child and node.assoc is not None:
            node.assoc = self._insert(node.assoc, pt, id_, dim + 1)

        return node

    # ------------------------------- delete --------------------------------
    def delete(self, coords: Sequence[float], id_: int) -> bool:
        """Deletes the point while ensuring consistency across the associated trees."""
        if self.size == 0:
            return False
        pt = np.asarray(coords, dtype=np.float64)
        deleted_main = [False]
        self.root = self._delete(self.root, pt, id_, 0, deleted_main)
        if deleted_main[0]:
            self.size -= 1
        return deleted_main[0]

    def _delete(self, node: Any, pt: np.ndarray, id_: int, dim: int, deleted_main: List[bool]) -> Any:
        if node is None:
            return None

        # Base Case 1: Fractional Cascading
        if dim == self.dims - 1:
            vals, ids = node
            mask = (ids == id_) & (vals == pt[dim])
            if np.any(mask):
                idx = np.argmax(mask)
                vals = np.delete(vals, idx)
                ids = np.delete(ids, idx)
                deleted_main[0] = True
            return (vals, ids)

        # Base Case 2: Bucket
        if isinstance(node, RangeNode) and node.bucket_mat is not None:
            coord_mask = np.all(node.bucket_mat == pt, axis=1)
            mask = (node.bucket_ids == id_) & coord_mask
            if np.any(mask):
                idx = np.argmax(mask)
                node.bucket_mat = np.delete(node.bucket_mat, idx, axis=0)
                node.bucket_ids = np.delete(node.bucket_ids, idx)
                deleted_main[0] = True
                if len(node.bucket_ids) == 0:
                    return None
            return node

        # Recursive Case
        child_deleted = [False]
        if node.left and pt[dim] <= node.left.max_val:
            node.left = self._delete(node.left, pt, id_, dim, child_deleted)
        
        if not child_deleted[0] and node.right and pt[dim] >= node.right.min_val:
            node.right = self._delete(node.right, pt, id_, dim, child_deleted)
            
        # Fallback 
        if not child_deleted[0]:
            if node.left:
                node.left = self._delete(node.left, pt, id_, dim, child_deleted)
            if not child_deleted[0] and node.right:
                node.right = self._delete(node.right, pt, id_, dim, child_deleted)

        if child_deleted[0]:
            deleted_main[0] = True
            if node.assoc is not None:
                dummy = [False]
                node.assoc = self._delete(node.assoc, pt, id_, dim + 1, dummy)

        # Pruning
        if node.left is None and node.right is None and node.bucket_mat is None:
            return None
            
        return node

    # ------------------------------- update --------------------------------
    def update(
        self,
        old_coords: Sequence[float],
        old_id: int,
        new_coords: Sequence[float],
        new_id: Optional[int] = None,
    ) -> bool:
        """Dynamically updates a point by removing the old one and inserting the new one."""
        if not self.delete(old_coords, old_id):
            return False
            
        final_id = new_id if new_id is not None else old_id
        self.insert(new_coords, final_id)
        return True

    # ------------------------------- range query --------------------------------
    def range_query(self, mins: Sequence[float], maxs: Sequence[float]) -> List[int]:
        mins_np = np.asarray(mins, dtype=np.float64)
        maxs_np = np.asarray(maxs, dtype=np.float64)
        result: List[int] = []
        
        if self.root is not None:
            self._query(self.root, mins_np, maxs_np, 0, result)
            
        return result

    def _query(self, node: Any, mins: np.ndarray, maxs: np.ndarray, dim: int, result: list) -> None:
        if dim == self.dims - 1:
            vals, ids = node
            start_idx = np.searchsorted(vals, mins[dim], side='left')
            end_idx = np.searchsorted(vals, maxs[dim], side='right')
            if start_idx < end_idx:
                result.extend(ids[start_idx:end_idx].tolist())
            return

        if maxs[dim] < node.min_val or mins[dim] > node.max_val:
            return

        if node.bucket_mat is not None:
            mat = node.bucket_mat
            mask = np.ones(len(mat), dtype=bool)
            for d in range(dim, self.dims):
                mask &= (mat[:, d] >= mins[d]) & (mat[:, d] <= maxs[d])
            result.extend(node.bucket_ids[mask].tolist())
            return

        if mins[dim] <= node.min_val and node.max_val <= maxs[dim]:
            self._query(node.assoc, mins, maxs, dim + 1, result)
            return

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