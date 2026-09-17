from dataclasses import dataclass
import math
import heapq

# =============================================================================
# Point
# =============================================================================

@dataclass(slots=True)
class Point:
    """
    Represents one point stored inside the multidimensional tree.

    `row_id` identifies the originating row (e.g. the DataFrame index),
    so the tree can index raw numeric data directly.
    """

    coordinates: tuple[float, ...]

    row_id: int

    def distance_to(self, other: "Point") -> float:
        """
        Returns the Euclidean distance between two points.
        """

        if len(self.coordinates) != len(other.coordinates):
            raise ValueError(
                "Points must have the same dimensionality."
            )

        return math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(
                    self.coordinates,
                    other.coordinates
                )
            )
        )

# =============================================================================
# Rectangle
# =============================================================================

EPS = 1e-9

@dataclass(slots=True)
class Rectangle:
    """
    Axis-aligned hyperrectangle in a multidimensional space.

    Parameters
    ----------
    centers : tuple[float, ...]
        Coordinates of the center of the hyperrectangle.

    half_sizes : tuple[float, ...]
        Half-size of the hyperrectangle along each dimension.
    """

    centers: tuple[float, ...]
    half_sizes: tuple[float, ...]

    def __post_init__(self) -> None:
        """
        Validates the dimensionality of the hyperrectangle.
        """

        if len(self.centers) != len(self.half_sizes):
            raise ValueError(
                "Centers and half-sizes must have the same dimensionality."
            )

        if len(self.centers) == 0:
            raise ValueError(
                "A rectangle must have at least one dimension."
            )

        if any(size < 0 for size in self.half_sizes):
            raise ValueError(
                "Half-sizes must be non-negative."
            )

    @property
    def dimension(self) -> int:
        """
        Returns the dimensionality of the hyperrectangle.
        """

        return len(self.centers)

    def contains(self, point: Point) -> bool:
        """
        Returns True if a point lies inside the hyperrectangle.
        """

        if len(point.coordinates) != self.dimension:
            return False

        for coordinate, center, half_size in zip(
            point.coordinates,
            self.centers,
            self.half_sizes
        ):
            if not (
                center - half_size - EPS
                <= coordinate
                <= center + half_size + EPS
            ):
                return False

        return True

    def intersects(self, other: "Rectangle") -> bool:
        """
        Returns True if two hyperrectangles intersect.
        """

        if self.dimension != other.dimension:
            return False

        for center_a, half_a, center_b, half_b in zip(
            self.centers,
            self.half_sizes,
            other.centers,
            other.half_sizes
        ):
            if (
                center_b - half_b
                > center_a + half_a + EPS
                or
                center_b + half_b
                < center_a - half_a - EPS
            ):
                return False

        return True

    def distance_to_point(self, point: Point) -> float:
        """
        Computes the minimum Euclidean distance between
        the hyperrectangle and a point.

        Used by kNN and similarity queries for pruning.
        """

        if len(point.coordinates) != self.dimension:
            raise ValueError(
                "Point and rectangle must have the same dimensionality."
            )

        squared_distance = 0.0

        for coordinate, center, half_size in zip(
            point.coordinates,
            self.centers,
            self.half_sizes
        ):
            distance = max(
                abs(coordinate - center) - half_size,
                0.0
            )

            squared_distance += distance * distance

        return math.sqrt(squared_distance)
    

# =============================================================================
# QuadTreeNode
# =============================================================================

class QuadTreeNode:
    """
    Represents one node of the multidimensional QuadTree.

    Every node owns a hyperrectangular region
    of the d-dimensional space.
    """

    def __init__(
        self,
        boundary: Rectangle,
        capacity: int = 50):

        if capacity <= 0:
            raise ValueError(
                "Capacity must be a positive integer."
            )

        self.boundary = boundary
        self.capacity = capacity

        self.points: list[Point] = []

        self._children: dict[
            tuple[int, ...],
            "QuadTreeNode"
        ] = {}

    def is_leaf(self) -> bool:
        """
        Returns True if the node has no children.
        """

        return not self._children

    def _get_child_for_point(self, point: Point) -> "QuadTreeNode":
        """
        Returns the child node that contains the given point,
        creating it lazily on first access.

        Each dimension contributes one binary decision:
        0 if the coordinate is less than or equal to the center,
        1 if the coordinate is greater than the center.

        Children are created on demand (rather than all 2^d at
        once) so that sparse/skewed regions of the space never
        pay for quadrants that end up empty.
        """

        if len(point.coordinates) != self.boundary.dimension:
            raise ValueError(
                "Point and boundary must have the same dimensionality."
            )

        child_index = tuple(
            1 if coordinate > center else 0
            for coordinate, center in zip(
                point.coordinates,
                self.boundary.centers
            )
        )

        if child_index not in self._children:
            child_centers = tuple(
                center + (
                    half_size / 2.0
                    if choice == 1
                    else -half_size / 2.0
                )
                for center, half_size, choice in zip(
                    self.boundary.centers,
                    self.boundary.half_sizes,
                    child_index
                )
            )

            child_half_sizes = tuple(
                half_size / 2.0
                for half_size in self.boundary.half_sizes
            )

            child_boundary = Rectangle(
                centers=child_centers,
                half_sizes=child_half_sizes
            )

            self._children[child_index] = QuadTreeNode(
                child_boundary,
                self.capacity
            )

        return self._children[child_index]


    def _insert_into_children(self, point: Point) -> bool:
        """
        Inserts a point into the appropriate child node.
        """

        child = self._get_child_for_point(point)

        if child is None:
            return False

        return child.insert(point)


    def insert(self, point: Point) -> bool:
        """
        Inserts a point into the QuadTree.
        """

        if not self.boundary.contains(point):
            return False

        if self.is_leaf() and len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if self.is_leaf():

            MIN_SIZE = 1e-9

            if any(
                half_size < MIN_SIZE
                for half_size in self.boundary.half_sizes
            ):
                self.points.append(point)
                return True

            old_points = self.points.copy()
            self.points.clear()

            for old_point in old_points:

                inserted = self._insert_into_children(old_point)

                if not inserted:
                    raise RuntimeError(
                        "Failed to redistribute an existing point."
                    )

        inserted = self._insert_into_children(point)

        if not inserted:
            raise RuntimeError(
                "Failed to insert point into any child."
            )

        return True


    def children(self) -> list["QuadTreeNode"]:
        """
        Returns all existing child nodes.
        """

        return list(self._children.values())


    def query(
        self,
        search_area: Rectangle,
        found_points: list[Point]
    ) -> None:
        """
        Performs a range query on this node.

        All points inside the search hyperrectangle are
        appended to found_points.
        """

        if not self.boundary.intersects(search_area):
            return

        for point in self.points:

            if search_area.contains(point):
                found_points.append(point)

        for child in self.children():
            child.query(
                search_area,
                found_points
            )


    def _knn_search(
        self,
        query_point: Point,
        k: int,
        heap: list[tuple[float, int, Point]]
    ) -> None:
        """
        Recursive helper for k-nearest neighbor search.
        """

        for point in self.points:

            distance = query_point.distance_to(point)

            if len(heap) < k:

                heapq.heappush(
                    heap,
                    (-distance, id(point), point)
                )

            else:

                current_farthest = -heap[0][0]

                if distance < current_farthest:

                    heapq.heapreplace(
                        heap,
                        (-distance, id(point), point)
                    )

        children = [
            (
                child.boundary.distance_to_point(query_point),
                child
            )
            for child in self.children()
        ]

        children.sort(key=lambda item: item[0])

        for min_distance, child in children:

            if len(heap) == k:

                current_farthest = -heap[0][0]

                if min_distance > current_farthest + EPS:
                    break

            child._knn_search(
                query_point,
                k,
                heap
            )

# =============================================================================
# QuadTree
# =============================================================================

class QuadTree:
    """
    Public interface of the Point QuadTree.
    """

    def __init__(
        self,
        boundary: Rectangle,
        capacity: int = 50
    ):

        self.capacity = capacity

        self.root = QuadTreeNode(
            boundary,
            capacity
        )

        self.size = 0


    def insert(self, point: Point) -> bool:
        """
        Inserts a point into the QuadTree.
        """

        inserted = self.root.insert(point)

        if inserted:
            self.size += 1

        return inserted


    def __len__(self) -> int:
        """
        Returns the number of stored points.
        """

        return self.size


    def __repr__(self) -> str:
        """
        String representation of the QuadTree.
        """

        return (
            f"QuadTree(size={self.size}, "
            f"capacity={self.capacity})"
        )


    def range_query(
    self,
    search_area: Rectangle
    ) -> list[Point]:
        """
        Returns all points inside the given rectangle.
        """

        found_points: list[Point] = []

        self.root.query(
            search_area,
            found_points
        )

        return found_points


    def knn(
    self,
    query_point: Point,
    k: int
    ) -> list[Point]:
        """
        Returns the k nearest neighbors of the query point.
        """

        if k <= 0:
            return []

        heap: list[tuple[float, int, Point]] = []

        self.root._knn_search(
            query_point,
            k,
            heap
        )

        result = sorted(
            heap,
            key=lambda item: -item[0]
        )

        return [
            point
            for _, _, point in result
        ]

# =============================================================================
# End-to-end pipeline : Quad Tree (phase 1) -> LSH (phase 2)
# =============================================================================
 
import time
from typing import Any, Dict, List, Tuple
 
import numpy as np
import pandas as pd
 
from kd_tree import KDAttribute as TreeAttribute, prepare_kd_matrix
from lsh import build_lsh_index
 
 
def run_quadtree_lsh_query(
    df: pd.DataFrame,
    tree_attributes: List[TreeAttribute],
    ranges: Dict[str, Tuple[float, float]],
    text_col: str | List[str],
    top_n: int = 10,
    num_perm: int = 100,
    bands: int = 20,
    min_shingles: int = 2,
) -> Dict[str, Any]:
    """
    Executes the end-to-end Quad Tree and LSH similarity pipeline, matching
    the signature/behavior of run_kd_lsh_query / run_rtree_lsh_query.
    """
    assert len(tree_attributes) <= 5, "Quad tree is restricted to k <= 5 in this project"
 
    # ---------- Phase 1 : build Quad Tree & run the range query ----------
    t0 = time.perf_counter()
    matrix, encodings = prepare_kd_matrix(df, tree_attributes)
    valid_mask = ~np.isnan(matrix).any(axis=1)
    ids = df.index.to_numpy()[valid_mask]
    matrix = matrix[valid_mask]
 
    if len(matrix) == 0:
        raise ValueError("No valid rows to index (all rows had missing values in the selected attributes).")
 
    # Bounding box of the actual data (raw units, no normalization needed:
    # the QuadTree lives in the same coordinate space as `ranges`).
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    centers = tuple((mn + mx) / 2 for mn, mx in zip(mins, maxs))
    half_sizes = tuple((mx - mn) / 2 + 1e-6 for mn, mx in zip(mins, maxs))
 
    boundary = Rectangle(centers=centers, half_sizes=half_sizes)
    tree = QuadTree(boundary)
 
    for row, row_id in zip(matrix, ids):
        tree.insert(Point(coordinates=tuple(row), row_id=int(row_id)))
 
    t_build = time.perf_counter() - t0
 
    # ---------- Range query ----------
    t0 = time.perf_counter()
 
    attr_cols = [a.column for a in tree_attributes]
    mins_q = [ranges[c][0] for c in attr_cols]
    maxs_q = [ranges[c][1] for c in attr_cols]
 
    q_centers = tuple((mn + mx) / 2 for mn, mx in zip(mins_q, maxs_q))
    q_half_sizes = tuple((mx - mn) / 2 for mn, mx in zip(mins_q, maxs_q))
    search_area = Rectangle(centers=q_centers, half_sizes=q_half_sizes)
 
    hit_points = tree.range_query(search_area)
    hit_ids = [p.row_id for p in hit_points]
 
    t_query = time.perf_counter() - t0
 
    subset = df.loc[hit_ids]
 
    # ---------- Phase 2 : LSH similarity search on the filtered subset ----------
    t0 = time.perf_counter()
    if len(hit_ids) > 1:
        lsh, _, skipped_low_info = build_lsh_index(
            subset, text_col, num_perm=num_perm, bands=bands, min_shingles=min_shingles
        )
        t_lsh_build = time.perf_counter() - t0
 
        t0 = time.perf_counter()
        top_pairs = lsh.top_n_pairs(n=top_n)
        t_lsh_query = time.perf_counter() - t0
    else:
        skipped_low_info = 0
        top_pairs = []
        t_lsh_build = 0.0
        t_lsh_query = 0.0
 
    return {
        "quad_tree_size": tree.size,
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