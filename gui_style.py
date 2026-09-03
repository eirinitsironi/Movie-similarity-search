import tkinter as tk
from tkinter import ttk


class Palette:
    # Basic backgrounds
    BG_APP = "#efe6b8"         
    BG_PANEL = "#efe6b8"        

    # Input surfaces (entry/combobox/listbox/spinbox)
    BG_INPUT = "#ede9d7"
    FG_INPUT = "#301e2d"
    BG_INPUT_DISABLED = "#d8d3bc"

    # Text
    FG_TEXT = "#1e2130"
    FG_TEXT_MUTED = "#9c9886"

    # Accent colors
    ACCENT = "#206B47"         
    ACCENT_HOVER = "#184F35"
    ACCENT_ACTIVE = "#c9c745"

    BORDER = "#d1c280"
    BORDER_LIGHT = "#aaa27f"

    ROW_ODD = "#87D5AF"
    ROW_EVEN = "#7DC3A1"
    ROW_SELECTED = "#A4E0C3"


FONT_FAMILY = "Segoe UI"


def _font(size=10, weight="normal"):
    return (FONT_FAMILY, size, weight)


# ======================================================================
# Apply theme
# ======================================================================

def apply_theme(root: tk.Tk) -> ttk.Style:
    p = Palette

    root.configure(bg=p.BG_APP)
    try:
        root.option_add("*Font", _font(10))
    except tk.TclError:
        pass

    style = ttk.Style(root)

    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # ---------------------------------------------------------------
    # General widgets
    # ---------------------------------------------------------------
    style.configure(
        ".",
        background=p.BG_APP,
        foreground=p.FG_TEXT,
        font=_font(10),
        borderwidth=0,
    )

    style.configure("TFrame", background=p.BG_APP)

    style.configure(
        "TLabel",
        background=p.BG_APP,
        foreground=p.FG_TEXT,
        font=_font(10),
    )

    # ---------------------------------------------------------------
    # LabelFrame 
    # ---------------------------------------------------------------
    style.configure(
        "TLabelframe",
        background=p.BG_PANEL,
        bordercolor=p.BORDER,
        relief="solid",
        borderwidth=1,
        padding=(10, 10),
    )
    style.configure(
        "TLabelframe.Label",
        background=p.BG_PANEL,
        foreground=p.ACCENT,
        font=_font(11, "bold"),
        padding=(6, 4),
    )

    # ---------------------------------------------------------------
    # Buttons
    # ---------------------------------------------------------------
    style.configure(
        "TButton",
        background=p.ACCENT,
        foreground="#87D5AF",
        font=_font(10, "bold"),
        padding=(14, 8),
        borderwidth=0,
        relief="flat",
        focuscolor=p.ACCENT,
    )
    style.map(
        "TButton",
        background=[("active", p.ACCENT_HOVER), ("pressed", p.ACCENT_ACTIVE),
                    ("disabled", p.BORDER)],
        foreground=[("disabled", p.FG_TEXT_MUTED)],
    )

    # ---------------------------------------------------------------
    # Entry
    # ---------------------------------------------------------------
    style.configure(
        "TEntry",
        fieldbackground=p.BG_INPUT,
        background=p.BG_INPUT,
        foreground=p.FG_INPUT,
        insertcolor=p.FG_INPUT,
        bordercolor=p.BORDER_LIGHT,
        lightcolor=p.BORDER_LIGHT,
        darkcolor=p.BORDER_LIGHT,
        borderwidth=1,
        relief="flat",
        padding=6,
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", p.ACCENT)],
        fieldbackground=[("disabled", p.BG_INPUT_DISABLED)],
    )

    # ---------------------------------------------------------------
    # Combobox
    # ---------------------------------------------------------------
    style.configure(
        "TCombobox",
        fieldbackground=p.BG_INPUT,
        background=p.BG_INPUT,
        foreground=p.FG_INPUT,
        arrowcolor=p.ACCENT,
        bordercolor=p.BORDER_LIGHT,
        borderwidth=1,
        padding=5,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", p.BG_INPUT), ("disabled", p.BG_INPUT_DISABLED)],
        foreground=[("readonly", p.FG_INPUT)],
        bordercolor=[("focus", p.ACCENT)],
    )
    root.option_add("*TCombobox*Listbox.background", p.BG_INPUT)
    root.option_add("*TCombobox*Listbox.foreground", p.FG_INPUT)
    root.option_add("*TCombobox*Listbox.selectBackground", p.ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", _font(10))

    # ---------------------------------------------------------------
    # Spinbox
    # ---------------------------------------------------------------
    style.configure(
        "TSpinbox",
        fieldbackground=p.BG_INPUT,
        background=p.BG_INPUT,
        foreground=p.FG_INPUT,
        arrowcolor=p.ACCENT,  
        arrowsize=16,          
        bordercolor=p.BORDER_LIGHT,
        lightcolor=p.BG_INPUT,     
        darkcolor=p.BG_INPUT,   
        borderwidth=1,
        relief="flat",             
        padding=5,
    )
    style.map(
        "TSpinbox",
        bordercolor=[("focus", p.ACCENT)],
        background=[("active", p.BORDER_LIGHT)],  
        arrowcolor=[("pressed", "#ffffff")]
    )

    # ---------------------------------------------------------------
    # Checkbutton
    # ---------------------------------------------------------------
    style.configure(
        "Toggle.Toolbutton",
        background=p.BG_INPUT,
        foreground=p.FG_INPUT,
        font=_font(10),
        padding=(10, 4),
        relief="flat",
        borderwidth=1,
    )
    style.map(
        "Toggle.Toolbutton",
        background=[("selected", p.ROW_SELECTED), ("active", p.BORDER_LIGHT)],
        foreground=[("selected", "#ffffff")],
    )

    # ---------------------------------------------------------------
    # Scrollbar
    # ---------------------------------------------------------------
    style.configure(
        "Vertical.TScrollbar",
        background=p.BORDER_LIGHT,
        troughcolor=p.BG_PANEL,
        bordercolor=p.BG_PANEL,
        arrowcolor=p.FG_TEXT,
        relief="flat",
        borderwidth=0,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", p.ACCENT), ("pressed", p.ACCENT_ACTIVE)],
    )

    # ---------------------------------------------------------------
    # Treeview 
    # ---------------------------------------------------------------
    style.configure(
        "Treeview",
        background=p.BG_APP,
        fieldbackground=p.BG_APP,
        foreground=p.FG_TEXT,
        rowheight=26,
        borderwidth=0,
        font=_font(10),
    )
    style.configure(
        "Treeview.Heading",
        background=p.ACCENT,
        foreground="#87D5AF",
        font=_font(10, "bold"),
        relief="flat",
        padding=(6, 6),
    )
    style.map(
        "Treeview.Heading",
        background=[("active", p.ACCENT_HOVER)],
    )
    style.map(
        "Treeview",
        background=[("selected", p.ROW_SELECTED)],
        foreground=[("selected", "#ffffff")],
    )



    return style


def style_listbox(listbox: tk.Listbox) -> None:
    p = Palette
    listbox.configure(
        background=p.BG_INPUT,
        foreground=p.FG_INPUT,
        selectbackground=p.ACCENT,
        selectforeground="#ffffff",
        activestyle="none",
        relief="flat",
        borderwidth=1,
        highlightthickness=1,
        highlightbackground=p.BORDER_LIGHT,
        highlightcolor=p.ACCENT,
        font=_font(10),
    )


def zebra_stripe_treeview(tree: ttk.Treeview) -> None:
    p = Palette
    tree.tag_configure("odd_row", background=p.ROW_ODD)
    tree.tag_configure("even_row", background=p.ROW_EVEN)
    for i, item in enumerate(tree.get_children("")):
        tag = "even_row" if i % 2 == 0 else "odd_row"
        tree.item(item, tags=(tag,))
