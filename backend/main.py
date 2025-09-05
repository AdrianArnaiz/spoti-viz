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
    Compute t-SNE (2D) for all rows and One-Class SVM scores based on the 'liked' subset.
    """
    df = DataStore.get_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Upload a CSV first.")

    # Choose features
    if params.feature_cols:
        missing = [c for c in params.feature_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Feature columns not found: {missing}")
        feat_df = df[params.feature_cols]
    else:
        feat_df = df.select_dtypes(include=[np.number])

    if feat_df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 numeric features for t-SNE.")

    # Drop rows with missing numeric features
    mask_no_na = feat_df.notna().all(axis=1)
    feat_df = feat_df.loc[mask_no_na]
    df_clean = df.loc[mask_no_na].reset_index(drop=True)
    X = feat_df.values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE
    # Ensure perplexity is valid: < (n_samples - 1)/3
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

    # One-Class SVM on liked subset (if found)
    liked_mask = detect_liked_mask(df_clean)
    if liked_mask.sum() == 0:
        liked_mask = np.ones(len(df_clean), dtype=bool)  # fallback: train on all

    X_pos = X_scaled[liked_mask]
    if X_pos.shape[0] < 10:
        # small liked set can be unstable; fallback to all
        X_pos = X_scaled

    ocsvm = OneClassSVM(kernel=params.kernel, gamma=params.gamma, nu=params.nu)
    ocsvm.fit(X_pos)
    scores = ocsvm.decision_function(X_scaled)
    preds = ocsvm.predict(X_scaled)  # 1 = inlier, -1 = outlier

    # Prepare payload (include a few common metadata columns if they exist)
    meta_cols = [c for c in ["track_name", "song_name", "name", "track", "artist", "artists", "playlist", "album"] if c in df_clean.columns]
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
            row[m] = None if pd.isna(df_clean.loc[i, m]) else str(df_clean.loc[i, m])
        payload.append(row)

    return {
        "n_samples": int(n_samples),
        "perplexity_used": float(max_perp),
        "ocsvm_params": {
            "nu": params.nu,
            "gamma": params.gamma,
            "kernel": params.kernel
        },
        "meta_columns": meta_cols,
        "points": payload
    }
