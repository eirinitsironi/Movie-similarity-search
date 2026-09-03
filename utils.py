import ast
from typing import Any, List
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the dataset. Handles specific formatting where the CSV uses 
    semicolons as separators and commas as decimal points.
    """
    return pd.read_csv(path, sep=";", decimal=",")


def extract_countries(cell: Any) -> List[str]:
    """Parses the origin_country cell, which is often a stringified list."""
    if isinstance(cell, list):
        return [str(c).strip() for c in cell]
    
    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                return [str(c).strip() for c in parsed]
            except (ValueError, SyntaxError):
                pass
        # Fallback: comma separated / bracket-stripped string
        return [c.strip() for c in s.strip("[]").replace("'", "").split(",") if c.strip()]
        
    return []