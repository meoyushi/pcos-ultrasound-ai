# PCOS Multimodal Predictor

An AI-powered web application that predicts **Polycystic Ovary Syndrome (PCOS)** risk using three complementary approaches:

| Mode | Input | Model |
|------|-------|-------|
| **Textual** | Clinical questionnaire (13 fields) | Random Forest (scikit-learn) — 0.84 accuracy, 0.89 ROC-AUC (5-fold CV) |
| **Ultrasound** | Ovarian ultrasound image | EfficientNetB0 transfer learning (TensorFlow) — **no validated accuracy; see [docs/03-findings.md](docs/03-findings.md)** |
| **Combined** | Both of the above | 60/40 weighted ensemble |

---

> ### ⚠️ Known limitation — the ultrasound dataset is leaked
>
> `data/train/Normal` and `data/train/PCOS` are two different image collections, not one
> cohort. They share **zero** image resolutions, and a single threshold on image *width*
> classifies the held-out test split at **88%** accuracy (majority baseline 59.5%).
>
> Any model trained here — including this one — scores near-perfectly by learning the
> acquisition source rather than the pathology. **No imaging accuracy figure from this
> dataset is meaningful.** The clinical Random Forest is unaffected.
>
> Verify with `python ml/ultrasound/audit_leakage.py`. Full analysis:
> [docs/03-findings.md](docs/03-findings.md).

---

## Folder Structure

```
pcos-ultrasound-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI app + CORS
│   │   ├── api/
│   │   │   ├── textual.py              # POST /api/predict/textual
│   │   │   ├── ultrasound.py           # POST /api/predict/ultrasound
│   │   │   └── combined.py             # POST /api/predict/combined
│   │   └── services/
│   │       ├── textual_service.py      # Random Forest singleton (13 form features)
│   │       └── ultrasound_service.py   # EfficientNetB0 singleton
│   ├── data/
│   │   └── PCOS_data_without_infertility.xlsx
│   ├── models/                          # Place pcos_efficientnet.h5 here
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx                    # React Router (6 routes)
│   │   ├── index.css                   # Design system
│   │   ├── components/
│   │   │   └── SymptomForm.jsx
│   │   ├── hooks/
│   │   │   └── useNeoPage.js
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Auth.jsx                # UI mockup only — not wired to a backend
│   │   │   ├── TextualPredict.jsx
│   │   │   ├── UltrasoundPredict.jsx
│   │   │   ├── CombinedPredict.jsx
│   │   │   └── Result.jsx
│   │   └── utils/
│   │       └── fields.js
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── vercel.json
│   └── .env.example
├── data/
│   ├── train/{Normal, PCOS}/           # Ultrasound training images
│   └── test/{Normal, PCOS}/            # Ultrasound test images
├── ml/
│   ├── textual/
│   │   ├── evaluate.py                 # cross-validates the served RF
│   │   └── metrics.json                # checked-in CV results
│   └── ultrasound/
│       ├── src/prepare_split.py        # builds a disjoint train/val/test index
│       ├── src/train.py                # EfficientNetB0 training script
│       └── splits/{train,val,test}.csv
├── docs/                                # 01-architecture, 02-models, 03-findings
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

> **Note:** The textual model trains on startup from the Excel dataset. The ultrasound model requires
> `pcos_efficientnet.h5` in `backend/models/`. If it is missing or fails to load, the ultrasound and
> combined endpoints return **503** and `GET /` reports `"status": "degraded"` — the API never returns
> a fabricated prediction.

### Frontend Setup

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

The dev server starts at `http://localhost:5173` and proxies `/api` requests to the backend.

---

## Training the Ultrasound Model

```bash
python ml/ultrasound/src/prepare_split.py   # build a disjoint train/val/test index
python ml/ultrasound/src/train.py
```

`prepare_split.py` hashes every image under `data/`, collapses duplicates (3846 files -> 1921
unique images), and writes a stratified 70/15/15 split to `ml/ultrasound/splits/*.csv`. This is
required: `data/train/` and `data/test/` as shipped are byte-identical copies of each other.

`train.py` then trains EfficientNetB0 with a frozen backbone, validates on the `val` split,
evaluates once on the held-out `test` split, writes `ml/ultrasound/metrics.json`, and saves the
model to `backend/models/pcos_efficientnet.h5`.

> **The `.h5` currently in the repo must be retrained before deployment.** It was trained with a
> double-normalisation bug (both training and serving divided by 255 before a network that
> normalises internally). That is now fixed in both places, which means the existing artifact no
> longer matches its own preprocessing.

### Evaluating the textual model

```bash
python ml/textual/evaluate.py
```

Cross-validates the exact model the API serves (5-fold stratified, 5 repeats) and writes
`ml/textual/metrics.json`. Current scores: **0.840 accuracy, 0.886 ROC-AUC, 0.731 sensitivity**
against a 0.673 majority-class baseline.

---

## Deployment

### Backend → Render

1. Create a new **Web Service** on [Render](https://render.com)
2. Set the **Root Directory** to `backend`
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Upload `pcos_efficientnet.h5` to `backend/models/` (or use Render Disk)

### Frontend → Vercel

1. Import the repo on [Vercel](https://vercel.com)
2. Set the **Root Directory** to `frontend`
3. **Build Command:** `npm run build`
4. **Output Directory:** `dist`
5. Add environment variable: `VITE_API_URL=https://your-backend.onrender.com`

---

## Environment Variables

| Variable | Location | Description |
|----------|----------|-------------|
| `VITE_API_URL` | `frontend/.env` | Backend API URL (e.g. `http://localhost:8000` for dev) |

---

## API Response Shape

All prediction endpoints return:

```json
{
  "success": true,
  "mode": "textual | ultrasound | combined",
  "prediction": 0 | 1,
  "label": "No PCOS" | "PCOS Detected",
  "confidence": 85.3,
  "pcos_probability": 72.1
}
```

If the ultrasound model is unavailable, `/api/predict/ultrasound` and `/api/predict/combined`
return **503** rather than a placeholder prediction:

```json
{ "detail": "Ultrasound model unavailable (...). Refusing to return a fabricated prediction." }
```

The `/api/predict/combined` endpoint also returns:

```json
{
  "breakdown": {
    "textual": { "label": "...", "confidence": ..., "pcos_probability": ... },
    "ultrasound": { "label": "...", "confidence": ..., "pcos_probability": ... }
  }
}
```

---

## Tech Stack

- **Frontend:** React 18, Vite, React Router v6, vanilla CSS
- **Backend:** FastAPI, scikit-learn (Random Forest), TensorFlow/Keras (EfficientNetB0)
- **Docs:** [`docs/01-architecture.md`](docs/01-architecture.md) · [`docs/02-models.md`](docs/02-models.md) · [`docs/03-findings.md`](docs/03-findings.md)
- **Design:** DM Serif Display + DM Sans fonts, sage/cream healthcare palette
