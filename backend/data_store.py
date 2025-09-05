\
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List

class DataStore:
    _df: Optional[pd.DataFrame] = None

    @classmethod
    def set_df(cls, df: pd.DataFrame) -> None:
        # Normalize column names a bit
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cls._df = df

    @classmethod
    def get_df(cls) -> Optional[pd.DataFrame]:
        return cls._df

def detect_label_candidates(df: pd.DataFrame) -> list[str]:
    """
    Try to guess columns suitable for coloring/labels in the UI.
    (categorical-ish or boolean)
    """
    candidates: list[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype == bool:
            candidates.append(col)
        elif s.dtype == object:
            # short unique sets -> likely categorical
            nunique = s.nunique(dropna=True)
            if 1 < nunique < min(50, len(df) // 5 + 1):
                candidates.append(col)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() < 20:
            candidates.append(col)
    # prioritize common spotify-ish columns by placing up front if present
    priority = [c for c in ["playlist", "liked", "is_liked", "in_library"] if c in candidates]
    rest = [c for c in candidates if c not in priority]
    return priority + rest

def detect_liked_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Heuristics to detect a 'liked' subset:
      - boolean column named liked/is_liked/in_library
      - a 'playlist' string column containing 'liked'
    Fallback: empty mask.
    """
    cols = [c.lower() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}

    for key in ["liked", "is_liked", "in_library"]:
        if key in colmap:
            try:
                return df[colmap[key]].astype(bool).values
            except Exception:
                pass

    if "playlist" in colmap:
        s = df[colmap["playlist"]].astype(str).str.lower()
        return s.str.contains("liked", na=False).values

    # could also try a column named 'source' with 'liked'
    if "source" in colmap:
        s = df[colmap["source"]].astype(str).str.lower()
        return s.str.contains("liked", na=False).values

    # fallback: empty
    return np.zeros(len(df), dtype=bool)
