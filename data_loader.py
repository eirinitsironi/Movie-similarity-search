from __future__ import annotations
from typing import Any

import ast
import pandas as pd

from quadtree import (
    Movie,
    Point,
    Rectangle,
    QuadTree
)

DIMENSION_ATTRIBUTES = [
    "popularity",
    "vote_average",
    "runtime",
    "release_year",
    "vote_count",
]

def parse_list(value: Any) -> list[str]:
    """
    Converts a string representation of a list into
    a Python list.
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

def extract_year(value: Any) -> int:
    if pd.isna(value):
        return 0
    value = str(value)
    if len(value) < 4:
        return 0
    try:
        return int(value[:4])
    except ValueError:
        return 0

def safe_float(value: Any) -> float:
    """
    Safely converts a value to float.
    Missing or invalid values become 0.0.
    """
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        value = value.replace(",", ".")
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def safe_int(value: Any) -> int:
    """
    Safely converts a value to int.
    Missing or invalid values become 0.
    """
    if pd.isna(value):
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def create_movie(row: dict) -> Movie:
    """
    Creates a Movie object from one row of the dataset.
    Χρησιμοποιούμε .get() για μέγιστη ασφάλεια έναντι KeyError.
    """
    return Movie(
        movie_id=safe_int(row.get("id")),
        title=str(row.get("title", "")),
        popularity=safe_float(row.get("popularity")),
        vote_average=safe_float(row.get("vote_average")),
        vote_count=safe_int(row.get("vote_count")),
        runtime=safe_float(row.get("runtime")),
        release_year=extract_year(row.get("release_date")),
        budget=safe_float(row.get("budget")),
        revenue=safe_float(row.get("revenue")),
    
        language=str(row.get("original_language", "")),
        countries=parse_list(row.get("origin_country")),
        genres=parse_list(row.get("genre_names")),
        production_companies=parse_list(row.get("production_company_names"))
    )


# =============================================================================
# Data Scaling (Min-Max Normalization)
# =============================================================================

class DataScaler:
    """
    Αποθηκεύει τα min/max της κάθε διάστασης για να κανονικοποιεί
    τα δεδομένα στο διάστημα [0, 1].
    """
    def __init__(self):
        self.mins: list[float] = []
        self.ranges: list[float] = []
        self.is_fitted: bool = False

    def fit(self, movies: list[Movie]) -> None:
        """
        Υπολογίζει τα min και max για κάθε διάσταση από το dataset.
        """
        if not movies:
            return

        dimension = len(DIMENSION_ATTRIBUTES)
        self.mins = [0.0] * dimension
        self.ranges = [1.0] * dimension
        
        for i, attr in enumerate(DIMENSION_ATTRIBUTES):
            values = [float(getattr(m, attr)) for m in movies]
            min_val = min(values)
            max_val = max(values)
            
            self.mins[i] = min_val
            range_val = max_val - min_val
            
            # Αποφυγή διαίρεσης με το 0 σε περίπτωση που όλες οι τιμές είναι ίδιες
            self.ranges[i] = range_val if range_val > 0 else 1.0
            
        self.is_fitted = True

    def normalize(self, movie: Movie) -> Point:
        """
        Μετατρέπει μια ταινία σε Point με κανονικοποιημένες συντεταγμένες [0, 1].
        """
        if not self.is_fitted:
            # Fallback: Αν δεν έχει γίνει fit, επιστρέφει τις αρχικές (raw) τιμές
            coordinates = tuple(
                float(getattr(movie, attr))
                for attr in DIMENSION_ATTRIBUTES
            )
            return Point(coordinates=coordinates, movie=movie)
            
        norm_coords = []
        for i, attr in enumerate(DIMENSION_ATTRIBUTES):
            raw_val = float(getattr(movie, attr))
            norm_val = (raw_val - self.mins[i]) / self.ranges[i]
            norm_coords.append(norm_val)
            
        return Point(
            coordinates=tuple(norm_coords),
            movie=movie
        )

# Δημιουργώ ένα global αντικείμενο scaler για να είναι διαθέσιμο παντού
scaler = DataScaler()

def movie_to_point(movie: Movie) -> Point:
    """
    Converts a Movie into a multidimensional Point.
    Πλέον επιστρέφει το σημείο κανονικοποιημένο (χρησιμοποιώντας τον scaler).
    """
    return scaler.normalize(movie)

def load_movies(csv_path: str) -> list[Movie]:
    """
    Loads all movies from the CSV file.
    """
    dataframe = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        encoding="latin1"
    )

    movies: list[Movie] = []

    # ΟΠΤΙΜΟΠΟΙΗΣΗ: Χρήση to_dict αντί για το αργό iterrows
    for row in dataframe.to_dict('records'):
        try:
            movie = create_movie(row)
            movies.append(movie)
        except Exception:
            # Skip corrupted rows.
            continue

    return movies

def compute_boundary(movies: list[Movie]) -> Rectangle:
    """
    Computes the minimum bounding hyperrectangle
    for all movies in the selected dimensions.
    """
    if not movies:
        raise ValueError("Movie list is empty.")

    points = [
        movie_to_point(movie)
        for movie in movies
    ]

    dimension = len(DIMENSION_ATTRIBUTES)

    centers = []
    half_sizes = []

    for i in range(dimension):
        values = [
            point.coordinates[i]
            for point in points
        ]

        min_value = min(values)
        max_value = max(values)

        center = (min_value + max_value) / 2
        half_size = (max_value - min_value) / 2

        half_size += 1e-6

        centers.append(center)
        half_sizes.append(half_size)

    return Rectangle(
        centers=tuple(centers),
        half_sizes=tuple(half_sizes)
    )

def build_quadtree(movies: list[Movie]) -> QuadTree:
    """
    Builds a multidimensional QuadTree from a list of movies.
    """
    boundary = compute_boundary(movies)
    tree = QuadTree(boundary)

    for movie in movies:
        point = movie_to_point(movie)
        tree.insert(point)

    return tree