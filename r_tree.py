"""
Project-1: Multi-dimensional Data Indexing and Similarity Query Processing
Scheme: R-tree + LSH (MinHash)

Implements the spatial search pipeline using a dynamic R-tree:
  - Phase 1 (R-Tree): Indexes numeric and encoded-categorical attributes 
    (k <= 5) using Minimum Bounding Rectangles (MBRs) to efficiently 
    resolve orthogonal range queries. Features Bottom-Up Bulk Loading 
    (Sort-Based) for extremely fast O(N log N) tree construction.
  - Phase 2 (LSH): Applies MinHash and Locality-Sensitive Hashing to textual 
    attributes of the filtered subset.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd

from lsh import build_lsh_index
from kd_tree import KDAttribute as TreeAttribute, prepare_kd_matrix

# ======================================================================
# Phase 1: R-TREE (k <= 5 dimensions)
# ======================================================================

def get_point_mbr(point: Sequence[float]) -> Tuple[Tuple[float, float], ...]:
    return tuple((float(val), float(val)) for val in point)

def get_enclosing_mbr(mbr1: Tuple[Tuple[float, float], ...], mbr2: Tuple[Tuple[float, float], ...]) -> Tuple[Tuple[float, float], ...]:
    return tuple((min(m1[0], m2[0]), max(m1[1], m2[1])) for m1, m2 in zip(mbr1, mbr2))

def mbr_area(mbr: Tuple[Tuple[float, float], ...]) -> float:
    area = 1.0
    for min_v, max_v in mbr:
        area *= max(0.0, max_v - min_v)
    return area

def mbr_enlargement(base_mbr: Tuple[Tuple[float, float], ...], new_mbr: Tuple[Tuple[float, float], ...]) -> float:
    enclosed = get_enclosing_mbr(base_mbr, new_mbr)
    return mbr_area(enclosed) - mbr_area(base_mbr)

def mbrs_intersect(mbr1: Tuple[Tuple[float, float], ...], mbr2: Tuple[Tuple[float, float], ...]) -> bool:
    for (min1, max1), (min2, max2) in zip(mbr1, mbr2):
        if min1 > max2 or max1 < min2:
            return False
    return True


class RTreeNode:
    def __init__(self, is_leaf: bool, max_entries: int):
        self.is_leaf = is_leaf
        self.max_entries = max_entries
        self.entries: List[Tuple[Tuple[Tuple[float, float], ...], Any]] = []

    def get_node_mbr(self) -> Tuple[Tuple[float, float], ...]:
        if not self.entries:
            raise ValueError("Empty node has no MBR")
        mbr = self.entries[0][0]
        for entry_mbr, _ in self.entries[1:]:
            mbr = get_enclosing_mbr(mbr, entry_mbr)
        return mbr

    def split_node(self) -> Tuple[Tuple[Tuple[float, float], ...], 'RTreeNode']:
        k = len(self.entries[0][0])
        best_dim = 0
        max_spread = -1.0
        
        for d in range(k):
            min_val = min(e[0][d][0] for e in self.entries)
            max_val = max(e[0][d][1] for e in self.entries)
            spread = max_val - min_val
            if spread > max_spread:
                max_spread = spread
                best_dim = d

        self.entries.sort(key=lambda e: (e[0][best_dim][0] + e[0][best_dim][1]) / 2.0)
        
        mid = len(self.entries) // 2
        new_node = RTreeNode(self.is_leaf, self.max_entries)
        new_node.entries = self.entries[mid:]
        self.entries = self.entries[:mid]
        
        return new_node.get_node_mbr(), new_node


class RTree:
    def __init__(self, max_entries: int = 16):
        self.max_entries = max_entries
        self.root = RTreeNode(is_leaf=True, max_entries=max_entries)
        self.size = 0
        
    def build(self, matrix: np.ndarray, ids: np.ndarray) -> None:
        """
        Μαζική Κατασκευή (Bottom-Up Bulk Loading).
        Εξαιρετικά γρήγορη κατασκευή ταξινομώντας τα δεδομένα.
        """
        self.size = len(matrix)
        if self.size == 0:
            return

        # 1. Ταξινόμηση
        order = np.argsort(matrix[:, 0])
        sorted_matrix = matrix[order]
        sorted_ids = ids[order]

        # 2. Φύλλα (Leaf Nodes)
        current_level_nodes = []
        for i in range(0, self.size, self.max_entries):
            chunk_mat = sorted_matrix[i : i + self.max_entries]
            chunk_ids = sorted_ids[i : i + self.max_entries]
            
            leaf = RTreeNode(is_leaf=True, max_entries=self.max_entries)
            for pt, p_id in zip(chunk_mat, chunk_ids):
                mbr = tuple((float(v), float(v)) for v in pt)
                leaf.entries.append((mbr, int(p_id)))
                
            current_level_nodes.append(leaf)

        # 3. Εσωτερικοί Κόμβοι (Bottom-Up)
        while len(current_level_nodes) > 1:
            next_level_nodes = []
            
            for i in range(0, len(current_level_nodes), self.max_entries):
                chunk_nodes = current_level_nodes[i : i + self.max_entries]
                
                parent = RTreeNode(is_leaf=False, max_entries=self.max_entries)
                for child_node in chunk_nodes:
                    parent.entries.append((child_node.get_node_mbr(), child_node))
                    
                next_level_nodes.append(parent)
                
            current_level_nodes = next_level_nodes

        self.root = current_level_nodes[0]

    def insert(self, item_id: int, coords: Tuple[float, ...]) -> None:
        point_mbr = get_point_mbr(coords)
        new_child = self._insert_recursive(self.root, point_mbr, item_id)
        
        if new_child is not None:
            new_child_mbr, new_node = new_child
            new_root = RTreeNode(is_leaf=False, max_entries=self.max_entries)
            new_root.entries.append((self.root.get_node_mbr(), self.root))
            new_root.entries.append((new_child_mbr, new_node))
            self.root = new_root

    def _insert_recursive(self, node: RTreeNode, point_mbr: Tuple[Tuple[float, float], ...], item_id: int):
        if node.is_leaf:
            node.entries.append((point_mbr, item_id))
            if len(node.entries) > node.max_entries:
                return node.split_node()
            return None

        best_idx = 0
        min_enl = float('inf')
        for i, (entry_mbr, _) in enumerate(node.entries):
            enl = mbr_enlargement(entry_mbr, point_mbr)
            if enl < min_enl:
                min_enl = enl
                best_idx = i
                
        best_mbr, best_child = node.entries[best_idx]
        new_split = self._insert_recursive(best_child, point_mbr, item_id)
        
        node.entries[best_idx] = (get_enclosing_mbr(best_mbr, point_mbr), best_child)
        
        if new_split is not None:
            new_child_mbr, new_node = new_split
            node.entries.append((new_child_mbr, new_node))
            if len(node.entries) > node.max_entries:
                return node.split_node()
        return None

    def range_query(self, mins: Sequence[float], maxs: Sequence[float]) -> List[int]:
        query_mbr = tuple((float(mi), float(ma)) for mi, ma in zip(mins, maxs))
        result: List[int] = []
        self._search_recursive(self.root, query_mbr, result)
        return result

    def _search_recursive(self, node: RTreeNode, query_mbr: Tuple[Tuple[float, float], ...], result: List[int]):
        for entry_mbr, child_or_data in node.entries:
            if mbrs_intersect(entry_mbr, query_mbr):
                if node.is_leaf:
                    result.append(child_or_data)
                else:
                    self._search_recursive(child_or_data, query_mbr, result)


# ======================================================================
# End-to-end pipeline : R-Tree (phase 1) -> LSH (phase 2)
# ======================================================================

def run_rtree_lsh_query(
    df: pd.DataFrame,
    tree_attributes: List[TreeAttribute],
    ranges: Dict[str, Tuple[float, float]],
    text_col: str | List[str],
    top_n: int = 10,
    num_perm: int = 100,
    bands: int = 20,
    min_shingles: int = 2,
) -> Dict[str, Any]:

    assert len(tree_attributes) <= 5, "R-tree is restricted to k <= 5 in this project"

    t0 = time.perf_counter()
    matrix, encodings = prepare_kd_matrix(df, tree_attributes)
    valid_mask = ~np.isnan(matrix).any(axis=1)
    ids = df.index.to_numpy()[valid_mask]
    matrix = matrix[valid_mask]

    tree = RTree(max_entries=16)
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
            "kd_build": t_build, #προς εσας κοριτσια να ξερετε πρεπει να το αλλαξουμε αυτο στο γκουι μαλλον αλλα τωρα κρατησα κδ ονομα
            "kd_range_query": t_query,
            "lsh_build": t_lsh_build,
            "lsh_query": t_lsh_query,
        },
    }