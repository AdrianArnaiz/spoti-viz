
# Spotify Taste Analysis — Web App

A modern, end‑to‑end web app that turns your Jupyter analysis into an interactive site with:

- **CSV upload**
- **t‑SNE + One‑Class SVM** projection dashboard (Tab 1)
- (Placeholders for) **Distributions** and **Playlists** dashboards
- Sleek **React + Tailwind** UI and **Plotly** interactivity
- **FastAPI** backend for data processing

https://github.com/ (create your own repo if you want)

---

## Project structure

```
spotify-viz-app/
├── backend/
│   ├── main.py               # FastAPI app: /api/upload and /api/tsne-ocsvm
│   └── data_store.py         # In‑memory DataFrame store + heuristics
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts        # Proxies /api → http://127.0.0.1:8000
│   ├── tailwind.config.cjs
│   ├── postcss.config.cjs
│   ├── tsconfig.json
│   └── src/
│       ├── styles.css
│       ├── main.tsx
│       ├── App.tsx           # Tabs + layout
│       ├── lib/api.ts        # axios client
│       └── components/
│           ├── FileUploader.tsx
│           ├── TsneOcsvmDashboard.tsx
│           ├── ControlsPanel.tsx
│           └── Plot.tsx
├── requirements.txt
├── .gitignore
└── sample_data/
    └── README.md
```

---

## Quickstart (local dev)

### 1) Backend (Python 3.10+)

```bash
cd spotify-viz-app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Backend runs at **http://127.0.0.1:8000**.

### 2) Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://127.0.0.1:5173** and proxies `/api` to the backend.

---

## Usage

1. Open the app, click **“Choose CSV…”**, and upload your Spotify export.
2. On the first tab, set **Perplexity**, **Iterations**, and **OC‑SVM** parameters.
3. Click **Run t‑SNE + OC‑SVM** to render the interactive scatter plot.
4. Hover for tooltips; zoom and pan are enabled (Plotly).
5. Use **Color by** to switch between **Inlier/Outlier** and **Playlist** coloring.

### CSV expectations

- Numeric features (e.g., `danceability`, `energy`, `valence`, `tempo`, etc.) are detected automatically.
- We try to detect a **“liked”** subset using:
  - `liked`, `is_liked`, or `in_library` boolean columns; or
  - `playlist` text containing “liked”.
- If none is found, OC‑SVM trains on all rows as a fallback.

> Tip: include helpful metadata columns like `track_name`, `artist`, `playlist` to enhance hover tooltips.

---

## How t‑SNE + OC‑SVM works here

- We standardize numeric features and compute a **2D t‑SNE** projection (init=`pca`, learning rate=`auto`).
- **OC‑SVM** is trained on the **liked** subset (if found) to model your “taste region.”
- Points are scored (`decision_function`); **inliers** are closer to your taste, **outliers** are further.

---

## Production build (optional)

1. Build the frontend:

```bash
cd frontend
npm run build
```

2. The FastAPI app will automatically serve `/frontend/dist` if present — you can now run:

```bash
cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

And visit **http://127.0.0.1:8000**.

---

## Extend the app

- Add endpoints (e.g., `/api/distributions`, `/api/similarity`) to mirror your notebook cells.
- Create new tabs with tailored dashboards:
  - **Distributions**: histograms/KDEs per playlist or liked status.
  - **Playlists**: t‑SNE by playlist, similarity matrices, recommendations.
- Persist datasets or allow multiple named sessions.
- Swap OC‑SVM for other novelty detectors (Isolation Forest, LOF).

---

## Notes

- This scaffold is intentionally modular and scalable: the backend holds data in `DataStore`, and the frontend uses composable components.
- For larger files or multi‑user deployments, swap in a database or object storage and user sessions.
