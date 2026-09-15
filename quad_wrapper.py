import time
import pandas as pd
import data_loader
from data_loader import create_movie, scaler, build_quadtree
from lsh import build_lsh_index

def run_quadtree_lsh_query(df, ranges, text_col, top_n, num_perm, bands, min_shingles):
    
    selected_attrs = list(ranges.keys())
    data_loader.DIMENSION_ATTRIBUTES = selected_attrs

    # 2. Μετατροπή του DataFrame σε λίστα από Movie objects
    movies = []
    for row in df.to_dict('records'):
        try:
            movies.append(create_movie(row))
        except Exception as e:
            # ΕΔΩ ΕΙΝΑΙ Η ΜΑΓΕΙΑ: Αντί να το κρύβουμε, το τυπώνουμε!
            print(f"Σφάλμα φόρτωσης ταινίας: {e}")
            continue

    if not movies:
        print("Προειδοποίηση: Η λίστα ταινιών είναι άδεια! Επιστροφή κενών αποτελεσμάτων.")
        return {
            "subset": df.set_index("id") if "id" in df.columns else df,
            "top_similar_pairs": [],
            "matched_count": 0,
            "skipped_low_info_text": 0,
            "timings_sec": {
                "tree_build": 0.0,
                "tree_query": 0.0,
                "lsh_build": 0.0,
                "lsh_query": 0.0
            }
        }

    # 3. PHASE 1: Κατασκευή QuadTree
    t0 = time.perf_counter()
    scaler.fit(movies)
    tree = build_quadtree(movies)
    t_build = time.perf_counter() - t0

    # 4. PHASE 1: Range Query
    t1 = time.perf_counter()
    
    # Κανονικοποίηση των Min-Max ορίων που έδωσε ο χρήστης
    min_norm, max_norm = [], []
    for i, attr in enumerate(selected_attrs):
        raw_min, raw_max = ranges[attr]
        
        rng = scaler.ranges[i] if scaler.ranges[i] > 0 else 1.0
        
        min_n = (raw_min - scaler.mins[i]) / rng
        max_n = (raw_max - scaler.mins[i]) / rng
        min_norm.append(min_n)
        max_norm.append(max_n)

    centers = tuple((mx + mn) / 2.0 for mx, mn in zip(max_norm, min_norm))
    half_sizes = tuple((mx - mn) / 2.0 for mx, mn in zip(max_norm, min_norm))
    
    search_area = type(tree.root.boundary)(centers=centers, half_sizes=half_sizes)
    filtered_points = tree.range_query(search_area)
    
    t_query = time.perf_counter() - t1

    filtered_movies = [p.movie for p in filtered_points]
    matched_count = len(filtered_movies)

    # 5. PHASE 2: Εκτέλεση LSH
    t2 = time.perf_counter()
    if matched_count > 1:
        
        # Μετάφραση του ονόματος για την κλάση Movie
        if text_col == "genre_names":
            movie_attr = "genres"
        else:
            movie_attr = "production_companies"
            
        # DataFrame με τα αποτελέσματα για το lsh.py
        lsh_data = {
            "id": [m.movie_id for m in filtered_movies],
            text_col: [getattr(m, movie_attr) for m in filtered_movies] 
        }

        df_lsh = pd.DataFrame(lsh_data).set_index("id")
        
        lsh_index, _, skipped = build_lsh_index(
            df=df_lsh, text_col=text_col, 
            num_perm=num_perm, bands=bands, min_shingles=min_shingles
        )
        t3 = time.perf_counter()
        
        top_pairs = lsh_index.top_n_pairs(n=top_n)
        t4 = time.perf_counter()
        
        lsh_build_time = t3 - t2
        lsh_query_time = t4 - t3
    else:
        top_pairs = []
        skipped = 0
        lsh_build_time = 0.0
        lsh_query_time = 0.0

    return {
        "subset": df.set_index("id") if "id" in df.columns else df,
        "top_similar_pairs": top_pairs,
        "matched_count": matched_count,
        "skipped_low_info_text": skipped,
        "timings_sec": {
            "tree_build": t_build,
            "tree_query": t_query,
            "lsh_build": lsh_build_time,
            "lsh_query": lsh_query_time
        }
    }