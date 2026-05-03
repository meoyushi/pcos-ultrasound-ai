# PCOS Multimodal Predictor

An AI-powered web application that predicts **Polycystic Ovary Syndrome (PCOS)** risk using three complementary approaches:

| Mode | Input | Model |
|------|-------|-------|
| **Textual** | Clinical questionnaire (14 fields) | Random Forest (scikit-learn) |
| **Ultrasound** | Ovarian ultrasound image | EfficientNetB0 (TensorFlow) |
| **Combined** | Both of the above | 60/40 weighted ensemble |

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
│   │       ├── textual_service.py      # Random Forest singleton
│   │       └── ultrasound_service.py   # EfficientNetB0 singleton
│   ├── data/
│   │   └── PCOS_data_without_infertility.xlsx
│   ├── models/                          # Place pcos_efficientnet.h5 here
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx                    # React Router (5 routes)
│   │   ├── index.css                   # Design system
│   │   ├── components/
│   │   │   └── SymptomForm.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
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
│   └── ultrasound/src/train.py         # EfficientNetB0 training script
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

> **Note:** The textual model trains on startup from the Excel dataset. The ultrasound model requires `pcos_efficientnet.h5` in `backend/models/` — if missing, it returns mock predictions.

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
python ml/ultrasound/src/train.py
```

This trains an EfficientNetB0 model using transfer learning on the images in `data/train/{Normal,PCOS}/` and evaluates on `data/test/{Normal,PCOS}/`. The trained model is saved to `backend/models/pcos_efficientnet.h5`.

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
- **Design:** DM Serif Display + DM Sans fonts, sage/cream healthcare palette
