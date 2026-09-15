"""
GUI for Project-1

Covers:
  1. Categorical Pre-Filters : original_language, origin_country, adult
     (dynamically populated from the ACTUAL unique values in the loaded CSV)
  2. Tree Configuration (Phase 1) : pick up to 5 numeric attributes, each
     with a Min/Max range, and which tree structure to use.
  3. Similarity Search (Phase 2 - LSH) : textual attribute + Top-N + min_shingles
  4. Results & Benchmarking : Top-N pairs table + build/query timings
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from kd_tree import KDAttribute as TreeAttribute, run_kd_lsh_query
from r_tree import run_rtree_lsh_query
from quad_wrapper import run_quadtree_lsh_query
from utils import load_dataset, extract_countries
import ctypes
import os
from gui_style import apply_theme, zebra_stripe_treeview, style_listbox


# no blurry scaling
if os.name == 'nt':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


NUMERIC_FIELDS = {
    "Release Year": "release_year",
    "Popularity": "popularity",
    "Vote Average": "vote_average",
    "Runtime": "runtime",
    "Budget": "budget",
    "Revenue": "revenue",
    "Vote Count": "vote_count",
}

TEXT_FIELDS = {
    "Genre Names": "genre_names",
    "Production Company Names": "production_company_names",
}

TREE_OPTIONS = ["k-d Tree", "Quad Tree", "Range Tree", "R-Tree"]
IMPLEMENTED_TREES = {"k-d Tree", "R-Tree", "Quad Tree"}

MAX_DIMS = 5


def get_unique_languages(df: pd.DataFrame) -> list:
    return sorted(df["original_language"].dropna().astype(str).unique().tolist())


def get_unique_countries(df: pd.DataFrame) -> list:
    codes = set()
    found = df["origin_country"].dropna().astype(str).str.findall(r"'([^']+)'")
    for lst in found:
        codes.update(c.strip() for c in lst if c.strip())
    return sorted(codes)


def build_category_mask(df: pd.DataFrame, languages: list, countries: list, adult: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    if languages:
        lang_set = {v.lower() for v in languages}
        mask &= df["original_language"].astype(str).str.strip().str.lower().isin(lang_set)

    if countries:
        country_set = set(countries)
        mask &= df["origin_country"].apply(
            lambda cell: bool(set(extract_countries(cell)) & country_set)
        )

    if adult == "Yes":
        mask &= df["adult"].astype(str).str.strip().str.upper() == "TRUE"
    elif adult == "No":
        mask &= df["adult"].astype(str).str.strip().str.upper() == "FALSE"
    # "All" -> no filter

    return mask


def validate_ranges(selected_fields: dict) -> tuple:
    """
    selected_fields: {label: (min_str, max_str)} for checked attributes only.
    Returns (tree_attributes, ranges_dict, error_message_or_None).
    """
    if not selected_fields:
        return None, None, "Select at least 1 numeric feature (k>=1)."
    if len(selected_fields) > MAX_DIMS:
        return None, None, f"You can select up to {MAX_DIMS} features (k<={MAX_DIMS})."

    tree_attributes = []
    ranges = {}
    for label, (min_s, max_s) in selected_fields.items():
        col = NUMERIC_FIELDS[label]
        try:
            lo = float(min_s.strip().replace(",", "."))
            hi = float(max_s.strip().replace(",", "."))
        except ValueError:
            return None, None, f"Invalid Min/Max values for '{label}'."
        if lo > hi:
            return None, None, f"Min > Max for '{label}'."
        tree_attributes.append(TreeAttribute(col, "numeric"))
        ranges[col] = (lo, hi)
    return tree_attributes, ranges, None


# ======================================================================
# GUI
# ======================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Movie Search System")
        self.geometry("980x980")
        apply_theme(self)

        self.df = None
        self.numeric_vars = {}      # label -> BooleanVar
        self.numeric_entries = {}   # label -> (min_entry, max_entry)
        self.result_queue = queue.Queue()

        self._build_load_section()
        self._build_categorical_section()
        self._build_tree_section()
        self._build_lsh_section()
        self._build_action_buttons()

    # ------------------------------- CSV loading --------------------------------
    def _build_load_section(self):
        frame = ttk.LabelFrame(self, text="Dataset")
        frame.pack(fill="x", padx=8, pady=6)

        self.path_var = tk.StringVar(value="data_movies_clean.csv")
        ttk.Entry(frame, textvariable=self.path_var, width=70).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(frame, text="Browse...", command=self._browse).grid(row=0, column=1, padx=4)
        ttk.Button(frame, text="Load Dataset", command=self._load_dataset_clicked).grid(row=0, column=2, padx=4)
        self.load_status = ttk.Label(frame, text="No dataset loaded.")
        self.load_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=4)

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _load_dataset_clicked(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Dataset", "Please enter the path to the CSV file.")
            return
        self.load_status.config(text="Loading... please wait.")
        threading.Thread(target=self._load_dataset_worker, args=(path,), daemon=True).start()

    def _load_dataset_worker(self, path):
        try:
            df = load_dataset(path)
            df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
            languages = get_unique_languages(df)
            countries = get_unique_countries(df)
            self.result_queue.put(("dataset_loaded", df, languages, countries))
        except Exception as e:
            self.result_queue.put(("dataset_error", str(e)))
        self.after(100, self._poll_queue)

    # ------------------------------- categorical --------------------------------
    def _build_categorical_section(self):
        frame = ttk.LabelFrame(self, text="1. Categorical Pre-Filters (empty selection = 'all')")
        frame.pack(fill="x", padx=8, pady=6)

        ttk.Label(frame, text="Original Language").grid(row=0, column=0)
        self.lang_listbox = tk.Listbox(frame, selectmode="multiple", height=6, exportselection=False)
        style_listbox(self.lang_listbox)
        self.lang_listbox.grid(row=1, column=0, padx=6, pady=4, sticky="ns")
        lang_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.lang_listbox.yview)
        lang_scroll.grid(row=1, column=1, sticky="ns")
        self.lang_listbox.config(yscrollcommand=lang_scroll.set)

        ttk.Label(frame, text="Origin Country").grid(row=0, column=2)
        self.country_listbox = tk.Listbox(frame, selectmode="multiple", height=6, exportselection=False)
        style_listbox(self.country_listbox)
        self.country_listbox.grid(row=1, column=2, padx=6, pady=4, sticky="ns")
        country_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.country_listbox.yview)
        country_scroll.grid(row=1, column=3, sticky="ns")
        self.country_listbox.config(yscrollcommand=country_scroll.set)

        ttk.Label(frame, text="Adult Content").grid(row=0, column=4, padx=6)
        self.adult_var = tk.StringVar(value="All")
        ttk.Combobox(frame, textvariable=self.adult_var, values=["All", "Yes", "No"],
                     state="readonly", width=10).grid(row=1, column=4, sticky="n", padx=6)

    # ------------------------------- tree config --------------------------------
    def _build_tree_section(self):
        frame = ttk.LabelFrame(
            self, text=f"2. Tree Configuration & Range Queries (Phase 1) — select up to {MAX_DIMS} features"
        )
        frame.pack(fill="x", padx=8, pady=6)

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Search Structure:").pack(side="left")
        self.tree_var = tk.StringVar(value=TREE_OPTIONS[0])
        ttk.Combobox(top, textvariable=self.tree_var, values=TREE_OPTIONS,
                     state="readonly", width=15).pack(side="left", padx=6)

        grid = ttk.Frame(frame)
        grid.pack(fill="x", padx=6, pady=4)
        ttk.Label(grid, text="Attributes").grid(row=0, column=0)
        ttk.Label(grid, text="Min").grid(row=0, column=1)
        ttk.Label(grid, text="Max").grid(row=0, column=2)

        for i, label in enumerate(NUMERIC_FIELDS, start=1):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(grid, text=label, variable=var,
                                  style="Toggle.Toolbutton",
                                  command=lambda l=label: self._enforce_max_dims(l))
            cb.grid(row=i, column=0, sticky="we", pady=3, padx=(0, 10))
            min_entry = ttk.Entry(grid, width=10)
            max_entry = ttk.Entry(grid, width=10)
            min_entry.grid(row=i, column=1, padx=4)
            max_entry.grid(row=i, column=2, padx=4)
            self.numeric_vars[label] = var
            self.numeric_entries[label] = (min_entry, max_entry)

    def _enforce_max_dims(self, just_toggled_label):
        checked = [l for l, v in self.numeric_vars.items() if v.get()]
        if len(checked) > MAX_DIMS:
            self.numeric_vars[just_toggled_label].set(False)
            messagebox.showwarning(
                "Warning", f"You can select up to {MAX_DIMS} features at once (k<={MAX_DIMS})."
            )

    # ------------------------------- lsh config --------------------------------
    def _build_lsh_section(self):
        frame = ttk.LabelFrame(self, text="3. Similarity Search (Phase 2 — LSH)")
        frame.pack(fill="x", padx=8, pady=6)

        ttk.Label(frame, text="Textual Attribute:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        self.text_field_var = tk.StringVar(value=list(TEXT_FIELDS.keys())[0])
        ttk.Combobox(frame, textvariable=self.text_field_var, values=list(TEXT_FIELDS.keys()),
                     state="readonly", width=28).grid(row=0, column=1, padx=6)

        ttk.Label(frame, text="Top-N:").grid(row=0, column=2, padx=6)
        self.top_n_var = tk.StringVar(value="3")
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.top_n_var, width=6).grid(row=0, column=3)

        ttk.Label(frame, text="Min Tags (Shingles):").grid(row=0, column=4, padx=6)
        self.min_shingles_var = tk.StringVar(value="2")
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.min_shingles_var, width=6).grid(row=0, column=5)

    # ------------------------------- actions --------------------------------
    def _build_action_buttons(self):
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=8, pady=6)
        ttk.Button(frame, text="Run Query", command=self._run_query).pack(side="left", padx=4)
        ttk.Button(frame, text="Run All Trees & Compare",
                   command=self._run_all_trees).pack(side="left", padx=4)
        self.run_status = ttk.Label(frame, text="")
        self.run_status.pack(side="left", padx=10)

    # ------------------------------- run query --------------------------------
    def _gather_selected_fields(self):
        selected = {}
        for label, var in self.numeric_vars.items():
            if var.get():
                min_entry, max_entry = self.numeric_entries[label]
                selected[label] = (min_entry.get(), max_entry.get())
        return selected

    def _run_query(self):
        if self.df is None:
            messagebox.showwarning("Dataset", "Please load the dataset first.")
            return

        tree_choice = self.tree_var.get()
        if tree_choice not in IMPLEMENTED_TREES:
            messagebox.showinfo(
                "Not Implemented",
                f"The '{tree_choice}' has not been implemented yet in this version.\n"
                f"Currently available: k-d Tree.",
            )
            return

        selected_fields = self._gather_selected_fields()
        tree_attributes, ranges, err = validate_ranges(selected_fields)
        if err:
            messagebox.showerror("Error", err)
            return

        try:
            top_n = int(self.top_n_var.get())
            min_shingles = int(self.min_shingles_var.get())
        except ValueError:
            messagebox.showerror("Error", "Top-N and min_shingles must be integers.")
            return

        languages = [self.lang_listbox.get(i) for i in self.lang_listbox.curselection()]
        countries = [self.country_listbox.get(i) for i in self.country_listbox.curselection()]
        adult = self.adult_var.get()
        text_col = TEXT_FIELDS[self.text_field_var.get()]

        self.run_status.config(text="Running... please wait.")
        threading.Thread(
            target=self._run_query_worker,
            args=(tree_attributes, ranges, text_col, top_n, min_shingles, languages, countries, adult),
            daemon=True,
        ).start()

    def _run_query_worker(self, tree_attributes, ranges, text_col, top_n, min_shingles, languages, countries, adult):
        try:
            mask = build_category_mask(self.df, languages, countries, adult)
            df_filtered = self.df[mask].copy().reset_index(drop=True)

            tree_name = self.tree_var.get()

            if tree_name == "k-d Tree":
                result = run_kd_lsh_query(
                    df=df_filtered,
                    kd_attributes=tree_attributes,
                    ranges=ranges,
                    text_col=text_col,
                    top_n=top_n,
                    num_perm=64,
                    bands=16,
                    min_shingles=min_shingles,
                )

            elif tree_name == "Quad Tree":
                result = run_quadtree_lsh_query(
                    df=df_filtered,
                    ranges=ranges,
                    text_col=text_col,
                    top_n=top_n,
                    num_perm=64,
                    bands=16,
                    min_shingles=min_shingles,
                )
            
            elif tree_name == "R-Tree":
                result = run_rtree_lsh_query(
                    df=df_filtered,
                    tree_attributes=tree_attributes, 
                    ranges=ranges,
                    text_col=text_col,
                    top_n=top_n,
                    num_perm=64,
                    bands=16,
                    min_shingles=min_shingles,
                )
            else:
                raise ValueError(f"Το δέντρο '{tree_name}' δεν έχει ενσωματωθεί ακόμα.")

            self.result_queue.put(("query_done", result, text_col, len(df_filtered), tree_name))
        except Exception as e:
            self.result_queue.put(("query_error", str(e)))
            
        self.after(100, self._poll_queue)

    def _run_all_trees(self):
        if self.df is None:
            messagebox.showwarning("Dataset", "Please load the dataset first.")
            return
        missing = ", ".join(t for t in TREE_OPTIONS if t not in IMPLEMENTED_TREES)
        messagebox.showinfo(
            "Run All Trees & Compare",
            f"Only the k-d Tree will be executed (the rest: {missing} have not been implemented yet).",
        )
        self.tree_var.set("k-d Tree")
        self._run_query()

    # ------------------------------- queue polling --------------------------------
    def _poll_queue(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                kind = msg[0]
                if kind == "dataset_loaded":
                    _, df, languages, countries = msg
                    self.df = df
                    self.lang_listbox.delete(0, "end")
                    for l in languages:
                        self.lang_listbox.insert("end", l)
                    self.country_listbox.delete(0, "end")
                    for c in countries:
                        self.country_listbox.insert("end", c)
                    self.load_status.config(
                        text=f"{len(df)} movies were loaded. "
                             f"{len(languages)} languages, {len(countries)} countries."
                    )
                elif kind == "dataset_error":
                    self.load_status.config(text="Error loading dataset.")
                    messagebox.showerror("Error", msg[1])
                elif kind == "query_done":
                    _, result, text_col, n_filtered, tree_name = msg
                    self._display_results(result, text_col, n_filtered, tree_name)
                    self.run_status.config(text="Completed!")
                elif kind == "query_error":
                    self.run_status.config(text="Error.")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass

# ------------------------------- results popup --------------------------------
    def _display_results(self, result, text_col, n_filtered, tree_name):
        results_window = tk.Toplevel(self)
        results_window.title("Results & Benchmarking")

        results_window.configure(bg="#efe6b8")

        frame = ttk.LabelFrame(results_window, text=f"Results & Benchmarking ({tree_name})")
        frame.pack(fill="both", expand=False, padx=8, pady=8)

        metrics_label = ttk.Label(frame, text="", justify="left")
        metrics_label.pack(anchor="w", padx=6, pady=4)

        num_results = len(result["top_similar_pairs"])

        display_height = max(3, min(num_results, 25))

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=False, padx=6, pady=4)

        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")

        columns = ("score", "movie_a", "text_a", "movie_b", "text_b")
        tree_view = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=display_height,
            yscrollcommand=y_scroll.set
        )

        def toggle_selection(event):
            item = tree_view.identify_row(event.y)
            if item in tree_view.selection():
                tree_view.selection_remove(item)
                return "break"
        
        tree_view.bind("<Button-1>", toggle_selection)

        y_scroll.config(command=tree_view.yview)

        y_scroll.pack(side="right", fill="y")
        tree_view.pack(side="left", fill="both", expand=True)

        for col, text, width in [
            ("score", "Score", 70), 
            ("movie_a", "Movie A", 220), 
            ("text_a", "Features A", 250),
            ("movie_b", "Movie B", 220), 
            ("text_b", "Features B", 250),
        ]:
            tree_view.heading(col, text=text, anchor="w")
            tree_view.column(col, width=width, minwidth=width, anchor="w")

        subset = result["subset"]
        for score, id1, id2 in result["top_similar_pairs"]:
            t1, g1 = subset.loc[id1, ["title", text_col]]
            t2, g2 = subset.loc[id2, ["title", text_col]]
            tree_view.insert("", "end", values=(f"{score:.3f}", t1, g1, t2, g2))

        zebra_stripe_treeview(tree_view)

        t = result["timings_sec"]
        
        t_build = t.get("tree_build", t.get("kd_build", 0.0))
        t_query = t.get("tree_query", t.get("kd_range_query", 0.0))
        
        # Calculation of total times
        total_build = t_build + t.get('lsh_build', 0.0)
        total_query = t_query + t.get('lsh_query', 0.0)
        
        metrics_label.config(text=(
            f"Data Statistics:\n"
            f"  • After categorical filters: {n_filtered} movies\n"
            f"  • Matched in {tree_name} range: {result.get('matched_count', 0)} movies\n"
            f"  • Skipped (low info text): {result.get('skipped_low_info_text', 0)} movies\n\n"
            f"Execution Timings:\n"
            f"  • Total Build Time: {total_build:.4f}s  ({tree_name}: {t_build:.4f}s  |  LSH: {t.get('lsh_build', 0.0):.4f}s)\n"
            f"  • Total Query Time: {total_query:.5f}s  ({tree_name}: {t_query:.5f}s  |  LSH: {t.get('lsh_query', 0.0):.5f}s)"
        ))

if __name__ == "__main__":
    app = App()
    app.mainloop()
