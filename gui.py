"""
GUI for Project-1 (Multi-Window Architecture)

Covers:
  1. Main Menu (Loads Dataset)
  2. Similarity Search Window (Phase 1 & Phase 2 LSH + Plots & Exports)
  3. Dynamic Tree Operations Window (Build, Insert, Delete, Update, k-NN, Bulk Benchmarking)
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import ctypes
import os
import csv
import random
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Imports από τα δικά σου αρχεία
from kd_tree import KDAttribute as TreeAttribute, run_kd_lsh_query, KDTree
from quadtree import run_quadtree_lsh_query, QuadTree, Point, Rectangle
from r_tree import run_rtree_lsh_query, RTree
from range_tree import run_rangetree_lsh_query, RangeTree
from utils import load_dataset, extract_countries
from gui_style import apply_theme, zebra_stripe_treeview, style_listbox

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
IMPLEMENTED_TREES = {"k-d Tree", "R-Tree", "Quad Tree", "Range Tree"}
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
    return mask

def validate_ranges(selected_fields: dict) -> tuple:
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
# 1. MAIN MENU
# ======================================================================
class MainMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Movie Search System - Main Menu")
        self.geometry("600x400")
        apply_theme(self)

        self.df = None
        self.languages = []
        self.countries = []

        self._build_ui()

    def _build_ui(self):
        title_label = ttk.Label(self, text="Movie Database Engine", font=("Segoe UI", 16, "bold"), foreground="#206B47")
        title_label.pack(pady=20)

        frame_load = ttk.LabelFrame(self, text="1. Load Dataset")
        frame_load.pack(fill="x", padx=40, pady=10)
        
        self.path_var = tk.StringVar(value="data_movies_clean.csv")
        ttk.Entry(frame_load, textvariable=self.path_var, width=40).pack(side="left", padx=10, pady=10)
        ttk.Button(frame_load, text="Browse...", command=self._browse).pack(side="left", padx=5)
        ttk.Button(frame_load, text="Load", command=self._load_dataset).pack(side="left", padx=5)
        
        self.load_status = ttk.Label(self, text="Status: Waiting for dataset...")
        self.load_status.pack(pady=5)

        frame_modules = ttk.LabelFrame(self, text="2. Select Module")
        frame_modules.pack(fill="x", padx=40, pady=10)

        self.btn_sim = ttk.Button(frame_modules, text="Similarity Search (Phase 1 & 2)", command=self.open_similarity)
        self.btn_sim.pack(fill="x", padx=20, pady=10)
        self.btn_sim.state(['disabled'])

        self.btn_ops = ttk.Button(frame_modules, text="Dynamic Tree Operations (Insert/Delete/Update/kNN)", command=self.open_operations)
        self.btn_ops.pack(fill="x", padx=20, pady=10)
        self.btn_ops.state(['disabled'])

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.path_var.set(path)

    def _load_dataset(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Dataset", "Please enter the path to the CSV file.")
            return
        self.load_status.config(text="Status: Loading... please wait.")
        threading.Thread(target=self._load_worker, args=(path,), daemon=True).start()

    def _load_worker(self, path):
        try:
            df = load_dataset(path)
            df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
            self.languages = get_unique_languages(df)
            self.countries = get_unique_countries(df)
            self.df = df
            self.after(0, self._on_load_success)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    def _on_load_success(self):
        self.load_status.config(text=f"Status: Loaded {len(self.df)} movies successfully!")
        self.btn_sim.state(['!disabled'])
        self.btn_ops.state(['!disabled'])

    def open_similarity(self):
        SimilarityWindow(self, self.df, self.languages, self.countries)

    def open_operations(self):
        OperationsWindow(self, self.df)


# ======================================================================
# 2. DYNAMIC OPERATIONS WINDOW
# ======================================================================
class OperationsWindow(tk.Toplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.title("Dynamic Tree Operations & Benchmarking")
        self.geometry("1000x850") 
        apply_theme(self)
        
        self.df = df
        self.active_tree = None
        self.tree_type = None 
        self.active_attrs = []
        self.attr_bounds = {}
        
        self.synthetic_data = [] 
        self.synthetic_id_counter = 1000000 
        
        self.numeric_vars = {}
        self.dynamic_entries = {}
        
        self._build_config_section()
        self._build_dynamic_inputs()
        self._build_console()

    def _build_config_section(self):
        frame = ttk.LabelFrame(self, text="1. Tree Setup")
        frame.pack(fill="x", padx=10, pady=5)

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=2)
        ttk.Label(top, text="Structure:").pack(side="left")
        self.tree_var = tk.StringVar(value=TREE_OPTIONS[0])
        ttk.Combobox(top, textvariable=self.tree_var, values=TREE_OPTIONS, state="readonly", width=15).pack(side="left", padx=6)
        
        mid = ttk.Frame(frame)
        mid.pack(fill="x", padx=6, pady=2)
        ttk.Label(mid, text="Select up to 5 dimensions:").pack(anchor="w")
        
        grid = ttk.Frame(mid)
        grid.pack(anchor="w", pady=2)
        for i, label in enumerate(NUMERIC_FIELDS):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(grid, text=label, variable=var, style="Toggle.Toolbutton", command=lambda l=label: self._enforce_max_dims(l))
            cb.grid(row=i//4, column=i%4, padx=5, pady=2)
            self.numeric_vars[label] = var

        ttk.Button(frame, text="Build Empty Tree / Reset", command=self._build_tree_action).pack(pady=5)

    def _enforce_max_dims(self, toggled_label):
        checked = [l for l, v in self.numeric_vars.items() if v.get()]
        if len(checked) > MAX_DIMS:
            self.numeric_vars[toggled_label].set(False)
            messagebox.showwarning("Limit", f"Max {MAX_DIMS} dimensions allowed.")

    def _build_dynamic_inputs(self):
        self.op_frame = ttk.LabelFrame(self, text="2. Single & Bulk Operations")
        self.op_frame.pack(fill="x", padx=10, pady=5)
        self.op_frame.pack_forget()

        self.inputs_container = ttk.Frame(self.op_frame)
        self.inputs_container.pack(fill="x", pady=2, padx=5)
        
        btn_frame = ttk.Frame(self.op_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Insert", command=self._insert_point).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Delete", command=self._delete_point).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Update", command=self._update_point).pack(side="left", padx=5)
        
        ttk.Label(btn_frame, text=" |  k=").pack(side="left", padx=(20,2))
        self.k_var = tk.StringVar(value="5")
        ttk.Entry(btn_frame, textvariable=self.k_var, width=5).pack(side="left")
        ttk.Button(btn_frame, text="k-NN Search", command=self._knn_search).pack(side="left", padx=5)

        ttk.Separator(self.op_frame, orient="horizontal").pack(fill="x", pady=5, padx=10)
        
        bulk_frame = ttk.Frame(self.op_frame)
        bulk_frame.pack(fill="x", pady=5, padx=5)
        ttk.Label(bulk_frame, text="Bulk Benchmarking (N =").pack(side="left")
        self.n_var = tk.StringVar(value="10000")
        ttk.Entry(bulk_frame, textvariable=self.n_var, width=8).pack(side="left")
        ttk.Label(bulk_frame, text=") : ").pack(side="left")

        ttk.Button(bulk_frame, text="Bulk Insert", command=self._bulk_insert).pack(side="left", padx=5)
        ttk.Button(bulk_frame, text="Bulk Delete", command=self._bulk_delete).pack(side="left", padx=5)
        ttk.Button(bulk_frame, text="Bulk k-NN", command=self._bulk_knn).pack(side="left", padx=5)

    def _build_console(self):
        frame = ttk.LabelFrame(self, text="Console / Output")
        frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.console = tk.Text(frame, height=7, bg="#1e2130", fg="#87D5AF", font=("Consolas", 10))
        self.console.pack(fill="both", expand=True, padx=5, pady=5)
        self.log("Ready. Select attributes and build the tree.")

    def log(self, message):
        self.console.insert("end", message + "\n")
        self.console.see("end")

    # --- ACTIONS ---
    def _build_tree_action(self):
        selected_labels = [l for l, v in self.numeric_vars.items() if v.get()]
        if not selected_labels:
            messagebox.showwarning("Error", "Select at least 1 attribute.")
            return

        self.active_attrs = [NUMERIC_FIELDS[l] for l in selected_labels]
        self.tree_type = self.tree_var.get()
        self.synthetic_data.clear()
        
        self.attr_bounds = {}
        for attr in self.active_attrs:
            self.attr_bounds[attr] = (self.df[attr].min(), self.df[attr].max())

        if self.tree_type == "Quad Tree":
            mins = [self.attr_bounds[a][0] for a in self.active_attrs]
            maxs = [self.attr_bounds[a][1] for a in self.active_attrs]
            centers = tuple((mn + mx) / 2 for mn, mx in zip(mins, maxs))
            half_sizes = tuple((mx - mn) / 2 + 1e-6 for mn, mx in zip(mins, maxs))
            self.active_tree = QuadTree(Rectangle(centers, half_sizes), capacity=50)
        elif self.tree_type == "k-d Tree":
            self.active_tree = KDTree(dims=len(self.active_attrs))
        elif self.tree_type == "R-Tree":
            self.active_tree = RTree(max_entries=16)
        elif self.tree_type == "Range Tree":
            self.active_tree = RangeTree(dims=len(self.active_attrs), leaf_size=4096)

        self.log(f"--- Built Empty {self.tree_type} for {len(self.active_attrs)} dimensions: {self.active_attrs} ---")
        
        for widget in self.inputs_container.winfo_children():
            widget.destroy()
        self.dynamic_entries.clear()

        ttk.Label(self.inputs_container, text="Movie ID (row_id):").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        ent_id = ttk.Entry(self.inputs_container, width=15)
        ent_id.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.dynamic_entries["row_id"] = ent_id

        for i, attr in enumerate(self.active_attrs):
            ttk.Label(self.inputs_container, text=f"{attr}:").grid(row=i+1, column=0, padx=5, pady=2, sticky="e")
            ent = ttk.Entry(self.inputs_container, width=15)
            ent.grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
            self.dynamic_entries[attr] = ent
            
            ttk.Label(self.inputs_container, text=f"New {attr} (For Update):").grid(row=i+1, column=2, padx=(30,5), pady=2, sticky="e")
            ent_new = ttk.Entry(self.inputs_container, width=15)
            ent_new.grid(row=i+1, column=3, padx=5, pady=2, sticky="w")
            self.dynamic_entries[f"new_{attr}"] = ent_new

        self.op_frame.pack(fill="x", padx=10, pady=5)

    # --- SINGLE OPERATIONS ---
    def _get_input_point(self, is_new=False):
        try:
            row_id = int(self.dynamic_entries["row_id"].get())
            coords = []
            for attr in self.active_attrs:
                key = f"new_{attr}" if is_new else attr
                val = self.dynamic_entries[key].get()
                if not val: return None 
                coords.append(float(val))
            return row_id, tuple(coords)
        except ValueError:
            return None

    def _insert_point(self):
        data = self._get_input_point()
        if not data: return
        row_id, coords = data
        
        if self.tree_type == "Quad Tree":
            self.active_tree.insert(Point(coords, row_id))
        elif self.tree_type == "k-d Tree":
            self.active_tree.insert(coords, row_id)
        elif self.tree_type == "R-Tree":
            self.active_tree.insert(row_id, coords)
        elif self.tree_type == "Range Tree":
            self.active_tree.insert(coords, row_id)
            
        self.log(f"SUCCESS: Inserted Movie {row_id} at {coords}")

    def _delete_point(self):
        data = self._get_input_point()
        if not data: return
        row_id, coords = data
        
        if self.tree_type == "Quad Tree":
            succ = self.active_tree.delete(Point(coords, row_id))
        elif self.tree_type == "k-d Tree":
            succ = self.active_tree.delete(coords, row_id)
        elif self.tree_type == "R-Tree":
            succ = self.active_tree.delete(coords, row_id)
        elif self.tree_type == "Range Tree":
            succ = self.active_tree.delete(coords, row_id)
            
        self.log(f"SUCCESS: Deleted Movie {row_id}." if succ else f"FAIL: Movie {row_id} not found.")

    def _update_point(self):
        old_data = self._get_input_point(is_new=False)
        new_data = self._get_input_point(is_new=True)
        if not old_data or not new_data: return
        old_id, old_coords = old_data
        new_id, new_coords = new_data
        
        if self.tree_type == "Quad Tree":
            succ = self.active_tree.update(Point(old_coords, old_id), Point(new_coords, old_id))
        elif self.tree_type == "k-d Tree":
            succ = self.active_tree.update(old_coords, old_id, new_coords, old_id)
        elif self.tree_type == "R-Tree":
            succ = self.active_tree.update(old_coords, old_id, new_coords, old_id)
        elif self.tree_type == "Range Tree":
            succ = self.active_tree.update(old_coords, old_id, new_coords, old_id)
            
        self.log(f"SUCCESS: Updated Movie {old_id}." if succ else "FAIL: Update failed.")

    def _knn_search(self):
        if self.tree_type == "Range Tree":
            self.log("INFO: Range Tree does not natively support k-NN queries.")
            return

        data = self._get_input_point()
        if not data: return
        _, coords = data
        try: k = int(self.k_var.get())
        except ValueError: return
        self.log(f"--- Searching {k}-NN for point {coords} ---")
        
        if self.tree_type == "Quad Tree":
            results = self.active_tree.knn(Point(coords, -1), k)
            for i, res in enumerate(results):
                dist = math.sqrt(sum((a-b)**2 for a,b in zip(coords, res.coordinates)))
                self.log(f"{i+1}. Movie ID {res.row_id} | Dist: {dist:.4f}")
        elif self.tree_type == "k-d Tree":
            results = self.active_tree.knn_query(coords, k)
            for i, (neg_dist, res_id) in enumerate(results):
                self.log(f"{i+1}. Movie ID {res_id} | Dist: {-neg_dist:.4f}")
        elif self.tree_type == "R-Tree":
            results = self.active_tree.knn_query(coords, k)
            for i, (dist, res_id) in enumerate(results):
                self.log(f"{i+1}. Movie ID {res_id} | Dist: {dist:.4f}")

    # --- BULK OPERATIONS ---
    def _bulk_insert(self):
        try: N = int(self.n_var.get())
        except ValueError: return
        
        pts = []
        for _ in range(N):
            coords = tuple(random.uniform(self.attr_bounds[a][0], self.attr_bounds[a][1]) for a in self.active_attrs)
            row_id = self.synthetic_id_counter
            self.synthetic_id_counter += 1
            pts.append((row_id, coords))

        t0 = time.perf_counter()
        if self.tree_type == "Quad Tree":
            for row_id, coords in pts:
                self.active_tree.insert(Point(coords, row_id))
        elif self.tree_type == "k-d Tree":
            for row_id, coords in pts:
                self.active_tree.insert(coords, row_id)
        elif self.tree_type == "R-Tree":
            for row_id, coords in pts:
                self.active_tree.insert(row_id, coords)
        elif self.tree_type == "Range Tree":
            for row_id, coords in pts:
                self.active_tree.insert(coords, row_id)
        t_total = time.perf_counter() - t0

        self.synthetic_data.extend(pts)
        self.log(f"BULK INSERT: {N} random points inserted into {self.tree_type} in {t_total:.5f} seconds.")

    def _bulk_delete(self):
        try: N = int(self.n_var.get())
        except ValueError: return
        
        if N > len(self.synthetic_data):
            N = len(self.synthetic_data)
        if N == 0:
            self.log("BULK DELETE: No data available. Please Bulk Insert first.")
            return

        to_delete = self.synthetic_data[-N:]
        self.synthetic_data = self.synthetic_data[:-N]

        t0 = time.perf_counter()
        if self.tree_type == "Quad Tree":
            for row_id, coords in to_delete:
                self.active_tree.delete(Point(coords, row_id))
        elif self.tree_type == "k-d Tree":
            for row_id, coords in to_delete:
                self.active_tree.delete(coords, row_id)
        elif self.tree_type == "R-Tree":
            for row_id, coords in to_delete:
                self.active_tree.delete(coords, row_id)
        elif self.tree_type == "Range Tree":
            for row_id, coords in to_delete:
                self.active_tree.delete(coords, row_id)
        t_total = time.perf_counter() - t0
        
        self.log(f"BULK DELETE: {N} points deleted from {self.tree_type} in {t_total:.5f} seconds.")

    def _bulk_knn(self):
        if self.tree_type == "Range Tree":
            self.log("BULK k-NN: Not executed. Range Tree does not support k-NN natively.")
            return

        try: N = int(self.n_var.get())
        except ValueError: return
        try: k = int(self.k_var.get())
        except ValueError: k = 5

        queries = []
        for _ in range(N):
            coords = tuple(random.uniform(self.attr_bounds[a][0], self.attr_bounds[a][1]) for a in self.active_attrs)
            queries.append(coords)

        t0 = time.perf_counter()
        if self.tree_type == "Quad Tree":
            for coords in queries:
                self.active_tree.knn(Point(coords, -1), k)
        elif self.tree_type == "k-d Tree":
            for coords in queries:
                self.active_tree.knn_query(coords, k)
        elif self.tree_type == "R-Tree":
            for coords in queries:
                self.active_tree.knn_query(coords, k)
        t_total = time.perf_counter() - t0
        
        self.log(f"BULK k-NN: Executed {N} queries (k={k}) on {self.tree_type} in {t_total:.5f} seconds.")


# ======================================================================
# 3. SIMILARITY SEARCH WINDOW (Restored Fully)
# ======================================================================
class SimilarityWindow(tk.Toplevel):
    def __init__(self, parent, df, languages, countries):
        super().__init__(parent)
        self.title("Similarity Search & Range Queries")
        self.geometry("1260x980")
        apply_theme(self)

        self.df = df
        self.languages = languages
        self.countries = countries

        self.numeric_vars = {}      
        self.numeric_entries = {}   
        self.result_queue = queue.Queue()

        self._build_categorical_section()
        self._build_tree_section()
        self._build_lsh_section()
        self._build_action_buttons()

        for l in self.languages:
            self.lang_listbox.insert("end", l)
        for c in self.countries:
            self.country_listbox.insert("end", c)

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

    def _build_tree_section(self):
        frame = ttk.LabelFrame(self, text=f"2. Tree Configuration & Range Queries (Phase 1) — select up to {MAX_DIMS} features")
        frame.pack(fill="x", padx=8, pady=6)
        top = ttk.Frame(frame)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Search Structure:").pack(side="left")
        self.tree_var = tk.StringVar(value=TREE_OPTIONS[0])
        ttk.Combobox(top, textvariable=self.tree_var, values=TREE_OPTIONS, state="readonly", width=15).pack(side="left", padx=6)
        grid = ttk.Frame(frame)
        grid.pack(fill="x", padx=6, pady=4)
        ttk.Label(grid, text="Attributes").grid(row=0, column=0)
        ttk.Label(grid, text="Min").grid(row=0, column=1)
        ttk.Label(grid, text="Max").grid(row=0, column=2)
        for i, label in enumerate(NUMERIC_FIELDS, start=1):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(grid, text=label, variable=var, style="Toggle.Toolbutton", command=lambda l=label: self._enforce_max_dims(l))
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
            messagebox.showwarning("Warning", f"You can select up to {MAX_DIMS} features at once (k<={MAX_DIMS}).")

    def _build_lsh_section(self):
        frame = ttk.LabelFrame(self, text="3. Similarity Search (Phase 2 — LSH)")
        frame.pack(fill="x", padx=8, pady=6)
        ttk.Label(frame, text="Textual Attribute:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        self.text_field_var = tk.StringVar(value=list(TEXT_FIELDS.keys())[0])
        ttk.Combobox(frame, textvariable=self.text_field_var, values=list(TEXT_FIELDS.keys()), state="readonly", width=28).grid(row=0, column=1, padx=6)
        ttk.Label(frame, text="Top-N:").grid(row=0, column=2, padx=6)
        self.top_n_var = tk.StringVar(value="3")
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.top_n_var, width=6).grid(row=0, column=3)
        ttk.Label(frame, text="Min Tags (Shingles):").grid(row=0, column=4, padx=6)
        self.min_shingles_var = tk.StringVar(value="2")
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.min_shingles_var, width=6).grid(row=0, column=5)
        ttk.Label(frame, text="Permutations:").grid(row=0, column=6, padx=6, pady=4, sticky="w")
        self.num_perm_var = tk.StringVar(value="64")
        ttk.Spinbox(frame, from_=1, to=1024, textvariable=self.num_perm_var, width=8).grid(row=0, column=7, sticky="w", padx=6)
        ttk.Label(frame, text="Bands:").grid(row=0, column=8, padx=6, sticky="w")
        self.bands_var = tk.StringVar(value="16")
        ttk.Spinbox(frame, from_=1, to=512, textvariable=self.bands_var, width=6).grid(row=0, column=9, sticky="w")

    def _build_action_buttons(self):
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=8, pady=6)
        ttk.Button(frame, text="Run Query", command=self._run_query).pack(side="left", padx=4)
        ttk.Button(frame, text="Run All Trees & Compare", command=self._run_all_trees).pack(side="left", padx=4)
        self.run_status = ttk.Label(frame, text="")
        self.run_status.pack(side="left", padx=10)

    def _gather_selected_fields(self):
        selected = {}
        for label, var in self.numeric_vars.items():
            if var.get():
                min_entry, max_entry = self.numeric_entries[label]
                selected[label] = (min_entry.get(), max_entry.get())
        return selected

    def _run_query(self):
        tree_choice = self.tree_var.get()
        if tree_choice not in IMPLEMENTED_TREES:
            messagebox.showinfo("Not Implemented", f"The '{tree_choice}' has not been implemented yet.\n")
            return
        selected_fields = self._gather_selected_fields()
        tree_attributes, ranges, err = validate_ranges(selected_fields)
        if err:
            messagebox.showerror("Error", err)
            return
        try:
            top_n = int(self.top_n_var.get())
            min_shingles = int(self.min_shingles_var.get())
            num_perm = int(self.num_perm_var.get())
            bands = int(self.bands_var.get())
        except ValueError:
            messagebox.showerror("Error", "Inputs must be integers.")
            return
        if num_perm % bands != 0:
            messagebox.showerror("Advanced LSH Error", "Permutations must be divisible by bands.")
            return

        languages = [self.lang_listbox.get(i) for i in self.lang_listbox.curselection()]
        countries = [self.country_listbox.get(i) for i in self.country_listbox.curselection()]
        adult = self.adult_var.get()
        text_col = TEXT_FIELDS[self.text_field_var.get()]

        self.run_status.config(text="Running... please wait.")
        threading.Thread(
            target=self._run_query_worker,
            args=(tree_attributes, ranges, text_col, top_n, min_shingles, num_perm, bands, languages, countries, adult),
            daemon=True,
        ).start()

    def _run_query_worker(self, tree_attributes, ranges, text_col, top_n, min_shingles, num_perm, bands, languages, countries, adult):
        try:
            mask = build_category_mask(self.df, languages, countries, adult)
            df_filtered = self.df[mask].copy().reset_index(drop=True)
            tree_name = self.tree_var.get()

            if tree_name == "k-d Tree":
                result = run_kd_lsh_query(df=df_filtered, kd_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            elif tree_name == "Quad Tree":
                result = run_quadtree_lsh_query(df=df_filtered, tree_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            elif tree_name == "R-Tree":
                result = run_rtree_lsh_query(df=df_filtered, tree_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            elif tree_name == "Range Tree":
                result = run_rangetree_lsh_query(df=df_filtered, tree_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            else:
                raise ValueError(f"Tree type '{tree_name}' is not implemented yet.")

            self.result_queue.put(("query_done", result, text_col, len(df_filtered), tree_name))
        except Exception as e:
            self.result_queue.put(("query_error", str(e)))
        self.after(100, self._poll_queue)

    def _run_all_trees(self):
        selected_fields = self._gather_selected_fields()
        tree_attributes, ranges, err = validate_ranges(selected_fields)
        if err:
            messagebox.showerror("Error", err)
            return
        try:
            top_n = int(self.top_n_var.get())
            min_shingles = int(self.min_shingles_var.get())
            num_perm = int(self.num_perm_var.get())
            bands = int(self.bands_var.get())
        except ValueError:
            return
        languages = [self.lang_listbox.get(i) for i in self.lang_listbox.curselection()]
        countries = [self.country_listbox.get(i) for i in self.country_listbox.curselection()]
        adult = self.adult_var.get()
        text_col = TEXT_FIELDS[self.text_field_var.get()]

        self.run_status.config(text="Running all trees... please wait.")
        threading.Thread(
            target=self._run_all_trees_worker,
            args=(tree_attributes, ranges, text_col, top_n, min_shingles, num_perm, bands, languages, countries, adult),
            daemon=True,
        ).start()

    def _run_all_trees_worker(self, tree_attributes, ranges, text_col, top_n, min_shingles, num_perm, bands, languages, countries, adult):
        try:
            mask = build_category_mask(self.df, languages, countries, adult)
            df_filtered = self.df[mask].copy().reset_index(drop=True)
            results = {}
            results["k-d Tree"] = run_kd_lsh_query(df=df_filtered, kd_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            results["Quad Tree"] = run_quadtree_lsh_query(df=df_filtered, tree_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            results["R-Tree"] = run_rtree_lsh_query(df=df_filtered, tree_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            results["Range Tree"] = run_rangetree_lsh_query(df=df_filtered, tree_attributes=tree_attributes, ranges=ranges, text_col=text_col, top_n=top_n, num_perm=num_perm, bands=bands, min_shingles=min_shingles)
            self.result_queue.put(("compare_done", results, len(df_filtered)))
        except Exception as e:
            self.result_queue.put(("query_error", str(e)))
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                kind = msg[0]
                if kind == "query_done":
                    _, result, text_col, n_filtered, tree_name = msg
                    self._display_results(result, text_col, n_filtered, tree_name)
                    self.run_status.config(text="Completed!")
                elif kind == "compare_done":
                    _, results_dict, n_filtered = msg
                    self._display_comparison(results_dict, n_filtered)
                    self.run_status.config(text="Comparison completed!")
                elif kind == "query_error":
                    self.run_status.config(text="Error.")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass

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
        tree_view = ttk.Treeview(tree_frame, columns=columns, show="headings", height=display_height, yscrollcommand=y_scroll.set)
        
        # Επαναφορά της επιλογής / αποεπιλογής γραμμής (toggle selection)
        def toggle_selection(event):
            item = tree_view.identify_row(event.y)
            if item in tree_view.selection():
                tree_view.selection_remove(item)
                return "break"
        tree_view.bind("<Button-1>", toggle_selection)

        y_scroll.config(command=tree_view.yview)
        y_scroll.pack(side="right", fill="y")
        tree_view.pack(side="left", fill="both", expand=True)

        for col, text, width in [("score", "Score", 70), ("movie_a", "Movie A", 220), ("text_a", "Features A", 250), ("movie_b", "Movie B", 220), ("text_b", "Features B", 250)]:
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

        def export_csv():
            filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")], title="Save Results")
            if not filepath: return
            try:
                with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["--- Data Statistics ---"])
                    writer.writerow(["Categorical filters applied", n_filtered])
                    writer.writerow([f"Matched in {tree_name}", result.get('matched_count', 0)])
                    writer.writerow(["Skipped", result.get('skipped_low_info_text', 0)])
                    writer.writerow([])
                    writer.writerow(["--- Execution Timings ---"])
                    writer.writerow(["Phase", "Total Time (s)", f"{tree_name} (s)", "LSH (s)"])
                    writer.writerow(["Build", f"{total_build:.4f}", f"{t_build:.4f}", f"{t.get('lsh_build', 0):.4f}"])
                    writer.writerow(["Query", f"{total_query:.5f}", f"{t_query:.5f}", f"{t.get('lsh_query', 0):.5f}"])
                    writer.writerow([])
                    writer.writerow(["--- Top N Similar Pairs ---"])
                    writer.writerow(["Score", "Movie A", "Features A", "Movie B", "Features B"])
                    for score, id1, id2 in result["top_similar_pairs"]:
                        tt1, gg1 = subset.loc[id1, ["title", text_col]]
                        tt2, gg2 = subset.loc[id2, ["title", text_col]]
                        writer.writerow([f"{score:.3f}", tt1, gg1, tt2, gg2])
                messagebox.showinfo("Export Successful", f"Results successfully exported to\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

        export_btn = ttk.Button(frame, text="Export to CSV", command=export_csv)
        export_btn.pack(pady=10)

    def _display_comparison(self, results_dict, n_filtered):
        comp_window = tk.Toplevel(self)
        comp_window.title("Exhaustive Tree Comparison")
        # Αυξήθηκε το μέγεθος του παραθύρου για να χωράνε άνετα και τα γραφήματα matplotlib
        comp_window.geometry("1080x750")
        comp_window.configure(bg="#efe6b8")

        frame = ttk.LabelFrame(comp_window, text="Performance Benchmarking")
        frame.pack(fill="both", expand=False, padx=8, pady=8)

        ttk.Label(frame, text=f"Initial subset after categorical filters: {n_filtered} movies").pack(anchor="w", padx=6, pady=4)

        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=False, padx=6, pady=4)
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
        display_height = len(results_dict)
        columns = ("tree", "matched", "skipped", "tree_build", "tree_query", "lsh_build", "lsh_query", "total_time")
        tree_view = ttk.Treeview(tree_frame, columns=columns, show="headings", height=display_height, yscrollcommand=y_scroll.set)
        
        # Επαναφορά της επιλογής / αποεπιλογής γραμμής στο πινακάκι σύγκρισης
        def toggle_selection(event):
            item = tree_view.identify_row(event.y)
            if item in tree_view.selection():
                tree_view.selection_remove(item)
                return "break"
        tree_view.bind("<Button-1>", toggle_selection)

        y_scroll.config(command=tree_view.yview)
        y_scroll.pack(side="right", fill="y")
        tree_view.pack(side="left", fill="both", expand=True)
        
        headers = [("tree", "Tree Structure", 120), ("matched", "Matched", 80), ("skipped", "Skipped", 80),
                   ("tree_build", "Tree Build (s)", 110), ("tree_query", "Tree Query (s)", 110),
                   ("lsh_build", "LSH Build (s)", 110), ("lsh_query", "LSH Query (s)", 110), ("total_time", "Total Time (s)", 110)]
        for col, text, width in headers:
            tree_view.heading(col, text=text, anchor="w")
            tree_view.column(col, width=width, minwidth=width, anchor="w")
            
        for tree_name, res in results_dict.items():
            t = res["timings_sec"]
            t_b = t.get("tree_build", t.get("kd_build", 0.0))
            t_q = t.get("tree_query", t.get("kd_range_query", 0.0))
            lsh_b = t.get("lsh_build", 0.0)
            lsh_q = t.get("lsh_query", 0.0)
            total = t_b + t_q + lsh_b + lsh_q
            tree_view.insert("", "end", values=(
                tree_name, 
                res.get("matched_count", 0), 
                res.get("skipped_low_info_text", 0),
                f"{t_b:.4f}", 
                f"{t_q:.5f}", 
                f"{lsh_b:.4f}", 
                f"{lsh_q:.5f}", 
                f"{total:.4f}"
            ))
            
        zebra_stripe_treeview(tree_view)

        # Επαναφορά του κουμπιού εξαγωγής CSV για τη Σύγκριση
        def export_comparison_csv():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
                title="Save Benchmarking Results"
            )
            if not filepath:
                return
            try:
                with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["--- Performance Benchmarking ---"])
                    writer.writerow(["Initial subset after categorical filters", n_filtered])
                    writer.writerow([])
                    writer.writerow([
                        "Tree Structure", "Matched", "Skipped", 
                        "Tree Build (s)", "Tree Query (s)", 
                        "LSH Build (s)", "LSH Query (s)", "Total Time (s)"
                    ])
                    
                    for tree_name, res in results_dict.items():
                        t = res["timings_sec"]
                        t_b = t.get("tree_build", t.get("kd_build", 0.0))
                        t_q = t.get("tree_query", t.get("kd_range_query", 0.0))
                        lsh_b = t.get("lsh_build", 0.0)
                        lsh_q = t.get("lsh_query", 0.0)
                        total = t_b + t_q + lsh_b + lsh_q
                        
                        writer.writerow([
                            tree_name,
                            res.get("matched_count", 0),
                            res.get("skipped_low_info_text", 0),
                            f"{t_b:.4f}",
                            f"{t_q:.5f}",
                            f"{lsh_b:.4f}",
                            f"{lsh_q:.5f}",
                            f"{total:.4f}"
                        ])
                        
                messagebox.showinfo("Export Successful", f"Benchmarking results successfully exported to\n{filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

        export_btn = ttk.Button(frame, text="Export Comparison to CSV", command=export_comparison_csv)
        export_btn.pack(pady=10)

        # Επαναφορά των Γραφημάτων (Matplotlib)
        plot_frame = ttk.Frame(comp_window)
        plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        trees = list(results_dict.keys())
        build_times = [results_dict[t]["timings_sec"].get("tree_build", results_dict[t]["timings_sec"].get("kd_build", 0.0)) for t in trees]
        query_times = [results_dict[t]["timings_sec"].get("tree_query", results_dict[t]["timings_sec"].get("kd_range_query", 0.0)) for t in trees]

        # Δημιουργία Figure με 2 subplots (1 γραμμή, 2 στήλες)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#efe6b8')

        bar_colors = ["#755050", "#8FA0AA", '#c9c745', "#634E73"]

        # Γράφημα 1: Build Times
        ax1.bar(trees, build_times, color=bar_colors[:len(trees)], edgecolor='black')
        ax1.set_title('Tree Build Time (seconds)', fontsize=11, fontweight='bold', color='#1e2130')
        ax1.set_ylabel('Time (s)')
        ax1.set_facecolor('#ede9d7')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Γράφημα 2: Query Times
        ax2.bar(trees, query_times, color=bar_colors[:len(trees)], edgecolor='black')
        ax2.set_title('Tree Query Time (seconds)', fontsize=11, fontweight='bold', color='#1e2130')
        ax2.set_ylabel('Time (s)')
        ax2.set_facecolor('#ede9d7')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = MainMenu()
    app.mainloop()