\
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import cosine_similarity

import io
import os

from .data_store import DataStore, detect_label_candidates, detect_liked_mask

app = FastAPI(title="Spotify Taste Analysis API", version="0.1.0")

# CORS (allow local dev frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",  # fine for local dev; tighten for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optionally serve built frontend if present (production)
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

class TSNEParams(BaseModel):
    perplexity: float = Field(30.0, ge=5.0, le=100.0)
    n_iter: int = Field(1000, ge=250, le=5000)
    learning_rate: str | float = "auto"
    random_state: Optional[int] = 42
    nu: float = Field(0.1, gt=0.0, lt=1.0)
    gamma: str | float = "scale"  # "scale", "auto", or float
    kernel: str = "rbf"
    label_preference: Optional[str] = None   # e.g., "liked", "playlist"
    feature_cols: Optional[List[str]] = None # if None, use all numeric
    train_on: str = Field("all", pattern="^(all|liked_auto)$")
    return_boundary: bool = True

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a CSV file and store it in memory.
    Returns summary info to drive the UI.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")

    # Store the dataframe in memory
    DataStore.set_df(df)

    # Numeric columns for modeling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Candidate label columns (string/bool)
    label_candidates = detect_label_candidates(df)

    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "label_candidates": label_candidates,
        "head_preview": df.head(10).to_dict(orient="records")
    }


@app.post("/api/tsne-ocsvm")
async def tsne_ocsvm(params: TSNEParams) -> Dict[str, Any]:
    """
    Compute t-SNE (2D) and One-Class SVM scores.
    Mirrors the notebook's preprocessing:
      - drop non-numeric metadata columns when present
      - convert 'Release Date' to year (YYYY) numeric
      - Z-score standardization
      - fit OC-SVM with gamma='scale' on chosen training set
    Returns points plus (optionally) the decision boundary grid.
    """
    df = DataStore.get_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Upload a CSV first.")

    # Notebook-style preprocessing
    work = df.copy()
    # columns commonly dropped in the notebook (ignore if missing)
    drop_cols = ['Track URI', 'Track Name', 'Album Name', 'Artist Name(s)',
                 'Explicit', 'Added By', 'Added At', 'Genres', 'Record Label']
    for c in drop_cols:
        if c in work.columns:
            work = work.drop(columns=[c])

    # Convert 'Release Date' to year
    if 'Release Date' in df.columns:
        # if it's like 'YYYY-MM-DD' or text, slice 0..4
        try:
            year = df['Release Date'].astype(str).str.slice(0, 4)
            work['Release Date'] = pd.to_numeric(year, errors='coerce')
        except Exception:
            pass

    # Choose features explicitly if provided, else all numeric
    if params.feature_cols:
        missing = [c for c in params.feature_cols if c not in df.columns and c not in work.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Feature columns not found: {missing}")
        feat_df = work[params.feature_cols] if set(params.feature_cols).issubset(work.columns) else df[params.feature_cols]
    else:
        feat_df = work.select_dtypes(include=[np.number])

    if feat_df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric features for t-SNE.")

    # Drop rows with NaNs in features
    mask_no_na = feat_df.notna().all(axis=1)
    feat_df = feat_df.loc[mask_no_na]
    df_clean = df.loc[mask_no_na].reset_index(drop=True)
    X = feat_df.values

    # Z-score standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE
    n_samples = X_scaled.shape[0]
    max_perp = max(5.0, min(params.perplexity, (n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=max_perp,
        learning_rate=params.learning_rate,
        n_iter=params.n_iter,
        init="pca",
        random_state=params.random_state,
        metric="euclidean",
    )
    tsne_coords = tsne.fit_transform(X_scaled)

    # Training set for OC-SVM
    if params.train_on == "liked_auto":
        liked_mask = detect_liked_mask(df_clean)
        if liked_mask.sum() == 0:
            X_train = tsne_coords
        else:
            X_train = tsne_coords[liked_mask]
    else:
        X_train = tsne_coords

    ocsvm = OneClassSVM(kernel=params.kernel, gamma=params.gamma, nu=params.nu)
    ocsvm.fit(X_train)
    scores = ocsvm.decision_function(tsne_coords)
    preds = ocsvm.predict(tsne_coords)  # 1=inlier, -1=outlier

    # Optional decision-boundary grid for contour overlay
    boundary = None
    if params.return_boundary:
        minx, maxx = float(tsne_coords[:,0].min()), float(tsne_coords[:,0].max())
        miny, maxy = float(tsne_coords[:,1].min()), float(tsne_coords[:,1].max())
        # pad a bit
        pad_x = (maxx - minx) * 0.08 if maxx > minx else 1.0
        pad_y = (maxy - miny) * 0.08 if maxy > miny else 1.0
        minx, maxx = minx - pad_x, maxx + pad_x
        miny, maxy = miny - pad_y, maxy + pad_y

        grid_n = 120  # trade-off between resolution and payload
        gx = np.linspace(minx, maxx, grid_n)
        gy = np.linspace(miny, maxy, grid_n)
        xx, yy = np.meshgrid(gx, gy)
        zz = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        boundary = {
            "x": gx.tolist(),
            "y": gy.tolist(),
            "z": zz.tolist()
        }

    # Metadata columns to bring back for hover (support notebook names)
    meta_cols = [c for c in [
        "Track Name", "Artist Name(s)", "Album Name", "playlist",
        "track_name", "artist", "artists", "album", "name"
    ] if c in df_clean.columns]

    payload = []
    for i, (x, y, s, p) in enumerate(zip(tsne_coords[:,0], tsne_coords[:,1], scores, preds)):
        row = {
            "id": int(i),
            "x": float(x),
            "y": float(y),
            "score": float(s),
            "inlier": bool(p == 1),
        }
        for m in meta_cols:
            v = df_clean.loc[i, m]
            row[m] = None if pd.isna(v) else str(v)
        payload.append(row)

    return {
        "n_samples": int(n_samples),
        "perplexity_used": float(max_perp),
        "ocsvm_params": {
            "nu": params.nu,
            "gamma": params.gamma,
            "kernel": params.kernel,
            "train_on": params.train_on
        },
        "meta_columns": meta_cols,
        "points": payload,
        "boundary": boundary
    }


class DistParams(BaseModel):
    feature: str
    group_by: str = Field("none", pattern="^(none|liked|playlist)$")
    bins: int = Field(30, ge=5, le=200)
    kde: bool = True
    top_groups: int = Field(8, ge=1, le=25)



@app.post("/api/distributions")
async def distributions(params: DistParams) -> Dict[str, Any]:
    """
    Compute histogram (and optional KDE) for a numeric feature,
    optionally grouped by liked or playlist (top N groups).
    """
    df = DataStore.get_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Upload a CSV first.")

    if params.feature not in df.columns:
        raise HTTPException(status_code=400, detail=f"Feature {params.feature} not in dataset.")

    # ensure numeric
    s = pd.to_numeric(df[params.feature], errors="coerce")
    mask_valid = s.notna()
    s = s[mask_valid]
    df_valid = df.loc[mask_valid]

    if s.empty:
        raise HTTPException(status_code=400, detail=f"No valid numeric values found for {params.feature}.")

    groups: list[tuple[str, pd.Series]] = []

    if params.group_by == "none":
        groups = [("All", s)]
    elif params.group_by == "liked":
        liked_mask = detect_liked_mask(df_valid)
        groups = [("Liked", s[liked_mask]), ("Other", s[~liked_mask])]
    elif params.group_by == "playlist" and "playlist" in df_valid.columns:
        top = (df_valid["playlist"].astype(str).value_counts().head(params.top_groups)).index.tolist()
        for name in top:
            grp = s[df_valid["playlist"].astype(str) == name]
            groups.append((str(name), grp))
    else:
        groups = [("All", s)]

    result_groups: list[Dict[str, Any]] = []
    # compute global bin edges for consistent overlay
    try:
        counts, bin_edges = np.histogram(s.values, bins=params.bins)
    except Exception:
        # fallback if bins fail
        bin_edges = np.linspace(float(s.min()), float(s.max()), params.bins + 1)

    grid_x = None
    if params.kde:
        grid_x = np.linspace(float(s.min()), float(s.max()), 200).reshape(-1, 1)

    for label, series in groups:
        vals = series.values.reshape(-1, 1)
        counts, _ = np.histogram(series.values, bins=bin_edges)
        out: Dict[str, Any] = {
            "label": label,
            "n": int(series.shape[0]),
            "hist": {
                "bin_edges": bin_edges.tolist(),
                "counts": counts.astype(int).tolist()
            }
        }
        if params.kde and series.shape[0] > 3:
            try:
                kde = KernelDensity(bandwidth=(series.std() * 1.06 * (series.shape[0] ** (-1/5))) if series.std() > 0 else 1.0)
                kde.fit(vals)
                log_dens = kde.score_samples(grid_x)
                dens = np.exp(log_dens)
                out["kde"] = {
                    "x": grid_x.flatten().tolist(),
                    "y": dens.tolist()
                }
            except Exception:
                pass
        result_groups.append(out)

    return {
        "feature": params.feature,
        "group_by": params.group_by,
        "groups": result_groups
    }



class PlaylistParams(BaseModel):
    top_groups: int = Field(12, ge=2, le=50)
    feature_cols: Optional[List[str]] = None



@app.post("/api/playlists/summary")
async def playlists_summary(params: PlaylistParams) -> Dict[str, Any]:
    df = DataStore.get_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
    if "playlist" not in df.columns:
        raise HTTPException(status_code=400, detail="No 'playlist' column found.")

    num_df = df.select_dtypes(include=[np.number])
    feats = params.feature_cols if params.feature_cols else num_df.columns.tolist()
    feats = [f for f in feats if f in num_df.columns]

    top = df["playlist"].astype(str).value_counts().head(params.top_groups).index.tolist()
    out_rows = []
    for name in top:
        mask = df["playlist"].astype(str) == name
        row = {
            "playlist": str(name),
            "count": int(mask.sum())
        }
        for f in feats[:10]:  # cap to avoid huge payloads
            row[f"avg_{f}"] = float(num_df.loc[mask, f].mean(skipna=True)) if f in num_df.columns else None
        out_rows.append(row)

    return {
        "features_used": feats[:10],
        "rows": out_rows
    }



@app.post("/api/playlists/similarity")
async def playlists_similarity(params: PlaylistParams) -> Dict[str, Any]:
    df = DataStore.get_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
    if "playlist" not in df.columns:
        raise HTTPException(status_code=400, detail="No 'playlist' column found.")

    num_df = df.select_dtypes(include=[np.number])
    feats = params.feature_cols if params.feature_cols else num_df.columns.tolist()
    feats = [f for f in feats if f in num_df.columns]
    if len(feats) == 0:
        raise HTTPException(status_code=400, detail="No numeric features available for similarity.")

    top = df["playlist"].astype(str).value_counts().head(params.top_groups).index.tolist()

    means = []
    for name in top:
        mask = df["playlist"].astype(str) == name
        v = num_df.loc[mask, feats].mean(skipna=True).values.reshape(1, -1)
        means.append(v)
    means_mat = np.vstack(means)

    sim = cosine_similarity(means_mat)
    return {
        "playlists": [str(x) for x in top],
        "features_used": feats,
        "similarity": sim.tolist()
    }

