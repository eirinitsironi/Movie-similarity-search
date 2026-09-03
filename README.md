# Movie Similarity Search

**Multi-dimensional Data Indexing and Similarity Query Processing**
*Two-phase pipeline: k-d Tree / Quad Tree / Range Tree / R-Tree indexing → LSH-based similarity search*

---

## Overview

This project implements a **two-phase search pipeline** for querying a large-scale movie
metadata dataset:

1. **Phase 1 — Multi-dimensional Indexing (k ≤ 5 attributes).**
   Numeric and label-encoded categorical attributes (e.g. `popularity`, `vote_average`,
   `runtime`, `release_year`) are indexed with a spatial data structure to efficiently
   resolve **orthogonal range queries**.
   Four interchangeable indexing schemes are implemented and benchmarked against each
   other: **k-d Tree, Quad Tree, Range Tree, R-Tree**.

2. **Phase 2 — Similarity Search via LSH.**
   The subset of movies returned by Phase 1 is then compared on a **textual attribute**
   (e.g. `genre_names`, `production_company_names`) using **MinHash + Locality-Sensitive
   Hashing (LSH)**, returning the **Top-N most similar pairs** without falling back to an
   expensive O(n²) pairwise comparison.

All algorithms (k-d tree, quad tree, range tree, R-tree, MinHash, banded LSH) are
implemented **from scratch**, without relying on external libraries such as
`scikit-learn` or `datasketch`, so that the underlying mechanics stay fully transparent
and inspectable.

## Example

> Find the Top-3 most similar movies by *genre* / *production company*, among movies
> released between 2000–2020, with popularity in [3, 6], vote_average in [3, 5],
> runtime in [30, 60], origin_country in {US, GB}, and original_language = "en".

## Dataset

[**Movies Metadata Cleaned Dataset (1900–2025)**](https://www.kaggle.com/datasets/mustafasayed1181/movies-metadata-cleaned-dataset-19002025) — TMDB-sourced, 946K+ movies.

| Column | Description |
|---|---|
| `id` | Unique movie identifier |
| `title` | Official movie title |
| `adult` | Boolean flag — adult content |
| `original_language` | ISO 639-1 language code |
| `origin_country` | List of production countries |
| `release_date` | Release date |
| `genre_names` | List of genres |
| `production_company_names` | List of production companies |
| `budget` | Reported production budget (USD) |
| `revenue` | Worldwide gross revenue (USD) |
| `runtime` | Duration (minutes) |
| `popularity` | TMDB popularity score |
| `vote_average` | Average user rating |
| `vote_count` | Number of votes |

## Running the project

```bash
python gui.py
```
1. Load `data_movies_clean.csv` (or browse to your CSV).
2. Apply categorical pre-filters (language, country, adult content) — optional.
3. Select up to 5 numeric attributes and their Min/Max ranges, and choose the indexing
   structure (k-d Tree, Quad Tree, Range Tree, or R-Tree).
4. Choose the textual attribute for similarity, Top-N, and minimum shingle count.
5. Click **Run Query** to see the Top-N most similar pairs plus build/query timings, or
   **Run All Trees & Compare** to execute the same query across all four structures.

## License / attribution
 
Movie metadata © TMDB, used under TMDB's Terms of Use. This project is not endorsed or
certified by TMDB. Code in this repository is original coursework implementation. 
