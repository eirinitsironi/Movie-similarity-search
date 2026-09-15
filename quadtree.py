from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import heapq

# =============================================================================
# Movie
# =============================================================================

@dataclass(slots=True)
class Movie:
    """
    Represents one movie from the dataset.
    Only the attributes required by the project are stored.
    """
    movie_id: int
    title: str
    popularity: float
    vote_average: float
    vote_count: int
    runtime: float
    release_year: int
    
    # ΠΡΟΣΘΗΚΗ ΤΩΝ ΠΕΔΙΩΝ ΠΟΥ ΕΛΕΙΠΑΝ
    budget: float
    revenue: float

    language: str
    countries: list[str]
    genres: list[str]
    production_companies: list[str]

# =============================================================================
# Point
# =============================================================================

@dataclass(slots=True)
class Point:
    """
    Represents one point stored inside the multidimensional tree.
    """
    coordinates: tuple[float, ...]
    movie: Movie

    def distance_to(self, other: "Point") -> float:
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Points must have the same dimensionality.")
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.coordinates, other.coordinates))
        )

# =============================================================================
# Rectangle
# =============================================================================

EPS = 1e-9

@dataclass(slots=True)
class Rectangle:
    """
    Axis-aligned hyperrectangle in a multidimensional space.
    """
    centers: tuple[float, ...]
    half_sizes: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.centers)

    def contains(self, point: Point) -> bool:
        if len(point.coordinates) != self.dimension:
            return False

        for coordinate, center, half_size in zip(point.coordinates, self.centers, self.half_sizes):
            if not (center - half_size - EPS <= coordinate <= center + half_size + EPS):
                return False
        return True

    def intersects(self, other: "Rectangle") -> bool:
        if self.dimension != other.dimension:
            return False

        for center_a, half_a, center_b, half_b in zip(self.centers, self.half_sizes, other.centers, other.half_sizes):
            if (center_b - half_b > center_a + half_a + EPS or 
                center_b + half_b < center_a - half_a - EPS):
                return False
        return True

    def distance_to_point(self, point: Point) -> float:
        if len(point.coordinates) != self.dimension:
            raise ValueError("Point and rectangle must have the same dimensionality.")

        squared_distance = 0.0
        for coordinate, center, half_size in zip(point.coordinates, self.centers, self.half_sizes):
            distance = max(abs(coordinate - center) - half_size, 0.0)
            squared_distance += distance * distance
        return math.sqrt(squared_distance)
    
# =============================================================================
# QuadTreeNode
# =============================================================================

class QuadTreeNode:
    """
    Represents one node of the multidimensional QuadTree.
    """
    def __init__(self, boundary: Rectangle, capacity: int = 50): # Χρησιμοποιούμε 50 για καλύτερη απόδοση σε 5D
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

        self.boundary = boundary
        self.capacity = capacity
        self.points: list[Point] = []
        self._children: dict[tuple[int, ...], "QuadTreeNode"] = {}
        
        # Lazy Initialization Flag
        self.is_divided = False 

    def is_leaf(self) -> bool:
        return not self.is_divided
    
    def subdivide(self) -> None:
        """
        Marks the node as divided. Actual children are generated lazily.
        """
        if self.boundary.dimension <= 0:
            raise ValueError("The tree must have at least one dimension.")
        self.is_divided = True

    def _get_child_for_point(self, point: Point) -> "QuadTreeNode":
        if len(point.coordinates) != self.boundary.dimension:
            raise ValueError("Point and boundary must have the same dimensionality.")

        child_index = tuple(
            1 if coordinate > center else 0
            for coordinate, center in zip(point.coordinates, self.boundary.centers)
        )

        # LAZY INITIALIZATION: Φτιάχνουμε το παιδί ΜΟΝΟ αν χρειαστεί
        if child_index not in self._children:
            child_centers = tuple(
                center + (half_size / 2.0 if choice == 1 else -half_size / 2.0)
                for center, half_size, choice in zip(self.boundary.centers, self.boundary.half_sizes, child_index)
            )
            child_half_sizes = tuple(half_size / 2.0 for half_size in self.boundary.half_sizes)
            
            child_boundary = Rectangle(centers=child_centers, half_sizes=child_half_sizes)
            self._children[child_index] = QuadTreeNode(child_boundary, self.capacity)

        return self._children[child_index]

    def _insert_into_children(self, point: Point) -> bool:
        child = self._get_child_for_point(point)
        if child is None:
            return False
        return child.insert(point)

    def insert(self, point: Point) -> bool:
        if not self.boundary.contains(point):
            return False

        if self.is_leaf() and len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if self.is_leaf():
            MIN_SIZE = 1e-9
            if any(half_size < MIN_SIZE for half_size in self.boundary.half_sizes):
                self.points.append(point)
                return True

            self.subdivide()
            old_points = self.points.copy()
            self.points.clear()

            for old_point in old_points:
                inserted = self._insert_into_children(old_point)
                if not inserted:
                    raise RuntimeError("Failed to redistribute an existing point.")

        inserted = self._insert_into_children(point)
        if not inserted:
            raise RuntimeError("Failed to insert point into any child.")
        return True

    def delete(self, point: Point) -> bool:
        if not self.boundary.contains(point):
            return False

        if self.is_leaf():
            for i, stored_point in enumerate(self.points):
                if stored_point.movie.movie_id == point.movie.movie_id:
                    del self.points[i]
                    return True
            return False

        # Αν το παιδί δεν έχει δημιουργηθεί ποτέ, το σημείο σίγουρα δεν υπάρχει
        child_index = tuple(
            1 if coordinate > center else 0
            for coordinate, center in zip(point.coordinates, self.boundary.centers)
        )
        if child_index not in self._children:
            return False

        child = self._children[child_index]
        deleted = child.delete(point)

        if deleted and self._can_merge():
            self._merge_children()
        return deleted

    def children(self) -> list["QuadTreeNode"]:
        return list(self._children.values())
    
    def _total_points_in_children(self) -> int:
        return sum(len(child.points) for child in self.children())

    def _can_merge(self) -> bool:
        if self.is_leaf():
            return False
        for child in self.children():
            if not child.is_leaf():
                return False
        return self._total_points_in_children() <= self.capacity

    def _merge_children(self) -> None:
        self.points = []
        for child in self.children():
            self.points.extend(child.points)
        self._children = {}
        self.is_divided = False # Επαναφορά του Flag!

    def query(self, search_area: Rectangle, found_points: list[Point]) -> None:
        if not self.boundary.intersects(search_area):
            return
        for point in self.points:
            if search_area.contains(point):
                found_points.append(point)
        for child in self.children():
            child.query(search_area, found_points)

    def _knn_search(self, query_point: Point, k: int, heap: list[tuple[float, int, Point]]) -> None:
        for point in self.points:
            distance = query_point.distance_to(point)
            if len(heap) < k:
                heapq.heappush(heap, (-distance, id(point), point))
            else:
                current_farthest = -heap[0][0]
                if distance < current_farthest:
                    heapq.heapreplace(heap, (-distance, id(point), point))

        children = [
            (child.boundary.distance_to_point(query_point), child)
            for child in self.children()
        ]
        children.sort(key=lambda item: item[0])

        for min_distance, child in children:
            if len(heap) == k:
                current_farthest = -heap[0][0]
                if min_distance > current_farthest + EPS:
                    break
            child._knn_search(query_point, k, heap)

    def _similarity_search(self, query_point: Point, radius: float, found_points: list[Point]) -> None:
        for point in self.points:
            if query_point.distance_to(point) <= radius:
                found_points.append(point)

        children = [
            (child.boundary.distance_to_point(query_point), child)
            for child in self.children()
        ]
        children.sort(key=lambda item: item[0])

        for min_distance, child in children:
            if min_distance > radius + EPS:
                break
            child._similarity_search(query_point, radius, found_points)

# =============================================================================
# QuadTree
# =============================================================================

class QuadTree:
    """
    Public interface of the Point QuadTree.
    """
    def __init__(self, boundary: Rectangle, capacity: int = 50):
        self.capacity = capacity
        self.root = QuadTreeNode(boundary, capacity)
        self.size = 0

    def insert(self, point: Point) -> bool:
        inserted = self.root.insert(point)
        if inserted:
            self.size += 1
        return inserted

    def delete(self, point: Point) -> bool:
        deleted = self.root.delete(point)
        if deleted:
            self.size -= 1
        return deleted

    def update(self, old_point: Point, new_point: Point) -> bool:
        if not self.delete(old_point):
            return False
        inserted = self.insert(new_point)
        if not inserted:
            self.insert(old_point)
            return False
        return True

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"QuadTree(size={self.size}, capacity={self.capacity})"

    def range_query(self, search_area: Rectangle) -> list[Point]:
        found_points: list[Point] = []
        self.root.query(search_area, found_points)
        return found_points

    def similarity_query(self, query_point: Point, radius: float) -> list[Point]:
        if radius < 0:
            return []
        found_points: list[Point] = []
        self.root._similarity_search(query_point, radius, found_points)
        return found_points

    def knn(self, query_point: Point, k: int) -> list[Point]:
        if k <= 0:
            return []
        heap: list[tuple[float, int, Point]] = []
        self.root._knn_search(query_point, k, heap)
        result = sorted(heap, key=lambda item: -item[0])
        return [point for _, _, point in result]