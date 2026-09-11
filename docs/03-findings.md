# 03 — Findings, Fixes, and What You Can Claim

The interview-prep document. What was broken, what was fixed, and the one thing that must not be
overstated.

---

## THE headline: the ultrasound dataset is leaked

`data/train/Normal` and `data/train/PCOS` are **two different image collections**, not one patient
cohort. They can be told apart from file metadata alone, without looking at a single pixel.

| | Normal | PCOS |
|---|---|---|
| Unique resolutions | 4 | 7 |
| Width range | 315-984 px | 225-343 px |
| Median file size | 55,328 B | 13,733 B |
| **Resolutions shared with the other class** | **0** | **0** |

Zero overlap. A single hand-fitted threshold on **image width** — no model, no pixels:

```
Rule fitted on train: predict PCOS if width <= 300
  HELD-OUT TEST accuracy  : 0.8824
  majority-class baseline : 0.5952
```

**88% from image width alone.** Any CNN trained here inherits that for free, plus JPEG and border
artifacts, which is why EfficientNetB0 scores a perfect 1.0000.

**How it surfaced:** after fixing preprocessing and building a disjoint split, a **frozen** ImageNet
backbone with an 82k-parameter head hit `val_accuracy: 1.0000` after **one epoch**. A frozen probe
cannot learn ovarian pathology in one epoch. Perfect accuracy that fast is a symptom, not a result.

This also explains the old pixel-variance heuristic's 0.961 AUC — greyscale std correlates with
resolution and JPEG quality. It was never detecting follicles.

```bash
python ml/ultrasound/audit_leakage.py   # writes ml/ultrasound/leakage_report.json
```

**Consequence:** no accuracy, AUC, sensitivity or specificity from this dataset measures PCOS
detection. Not the old model's, not the heuristic's, not the retrained model's. A clean train/test
split does not help when the label is encoded in the file format itself.

**To make the imaging claim real:** source a corpus where both classes come from the same scanner
and export pipeline, and run `audit_leakage.py` on it before trusting any number.

---

## What you can and cannot say

### Safe — every word verifiable

> Architected a 3-mode PCOS risk-assessment pipeline: a Random Forest on 13 clinical and lifestyle
> features (**0.84 accuracy, 0.89 ROC-AUC**, 5-fold CV, n=541), an EfficientNetB0 transfer-learning
> ultrasound classifier, and a 60/40 weighted probability ensemble.

> Shipped a FastAPI + React application across 3 prediction modes (clinical, ultrasound, combined),
> with a Vercel-hosted SPA and a Render-deployed inference API.

Strong optional third bullet — the best engineering story in the project:

> Audited the ultrasound corpus and found the class folders separable by image resolution alone
> (zero overlap; a width-threshold baseline scored 88% held-out), invalidating a 100%-accuracy
> result before it shipped.

Or, on the clinical side:

> Diagnosed a train/serve feature mismatch — the model was fit on 41 features but served 13 with
> medians imputed — and **restored sensitivity from 0.26 to 0.73** by retraining on the deployed
> feature set.

### Not safe

| Do not say | Why |
|---|---|
| Any ultrasound accuracy / AUC number | Dataset is leaked. Indefensible. |
| "0.85 accuracy" for the clinical model | Measured value is **0.8403**, so 0.84. |
| "94% accuracy" | Was on the old landing page with no backing artifact. Now removed. |
| "Fine-tuned EfficientNetB0" *(of the original)* | The original froze the backbone and never unfroze it. The **current** code does fine-tune, so it is true of the current version only. |
| "14 clinical parameters" | The form has 13. |

### Questions to expect

- *"Why does the clinical model use only 13 features when the dataset has 41?"* — Because the form
  collects 13. Training on 41 and imputing the rest dropped sensitivity to 0.26; the follicle
  counts, carrying ~31% of importance, were pinned to a constant.
- *"Your imaging model gets 100%. What is your baseline?"* — Image width alone gets 88%. The dataset
  is leaked; that is why I do not quote an imaging metric.
- *"How do you know the old CNN was not working?"* — It emitted 0.4218-0.4267 for every image, a
  0.005-wide range, so the fallback heuristic fired 100% of the time.
- *"Is 0.89 AUC good for PCOS screening?"* — Reasonable for a symptom questionnaire with no labs or
  imaging, against a 0.673 majority baseline. It is cross-validation on one 541-patient
  single-source dataset with no external validation cohort.

---

## Fixes applied

**1. Clinical model — train/serve parity.** Trains on the 13 collected fields plus derived BMI
instead of 41 columns with 28 medians. Sensitivity **0.259 -> 0.731**, accuracy 0.747 -> 0.840.
Missing/non-numeric fields and non-positive height or weight now return 400 instead of being
silently imputed.

**2. Imaging — EfficientNetB0 is actually used.** The `(0.1, 0.9)` override that replaced the CNN
with a pixel-variance heuristic is off by default. Verified: every response now carries
`used_pixel_std_fallback: false`.

**3. Double normalisation removed.** `/255` gone from both `train.py` and `_preprocess` — Keras'
EfficientNetB0 normalises internally and expects 0-255. This was the likely cause of the model
"collapse" the old comments described.

**4. Real image split.** `prepare_split.py` — 3,846 files to 1,921 unique, stratified 70/15/15,
asserted disjoint. Validation and test are now genuinely held out, and no pixels are copied.

**5. Silent failure became loud failure.** A missing or unloadable model used to return
`"No PCOS", confidence 50.0` with HTTP 200 — indistinguishable from a real negative — while `GET /`
still reported `healthy`. Now `predict()` raises `ModelUnavailableError`, both imaging routes return
**503**, and `GET /` reports `degraded` with a per-model readiness map. The combined route no longer
blends a fabricated 50% into the ensemble.

**6. Metrics are checked in.** `ml/textual/evaluate.py` writes `ml/textual/metrics.json`; `train.py`
writes `ml/ultrasound/metrics.json` (auto-stamped with the leakage verdict); `audit_leakage.py`
writes `ml/ultrasound/leakage_report.json`. Every number quoted anywhere traces to a regenerable file.

**7. Smaller corrections.** MIME check moved before inference in the combined route;
`ALLOWED_TYPES` de-duplicated; handlers no longer echo raw exception strings; the `debug` block
leaking absolute paths and `cwd` removed; `parseFloat(val) || 0` replaced with `toFeaturePayload()`
so a blank Age is not sent as a measured zero; landing-page claims corrected (94% to 0.89 AUC,
14 to 13 parameters, "fine-tuned" to "transfer-learned"); dead `center` crop, unused `Dropout`
import and unused `base` return removed; dependency floors added with the Keras 3 requirement
documented.

---

## Still open

| Issue | Where | Severity |
|---|---|---|
| Imaging dataset is leaked — no valid metric obtainable | `data/` | **Blocking for any imaging claim** |
| `/auth` is a non-functional mockup, and its copy advertises persistence features that do not exist | [`Auth.jsx`](../frontend/src/pages/Auth.jsx) | Medium |
| CORS is `allow_origins=["*"]` | [`main.py`](../backend/app/main.py) | Medium — fix before public deploy |
| No upload size limit; MIME check trusts a client header | [`ultrasound.py`](../backend/app/api/ultrasound.py) | Medium — memory exhaustion vector |
| No tests, no CI, no linting | whole repo | Medium |
| Random Forest refits on every worker boot; no persistence | [`textual_service.py`](../backend/app/services/textual_service.py) | Low |
| Confidence is uncalibrated (no Platt/isotonic scaling) | both services | Low, but it is shown as a percentage |
| `*.h5` is gitignored, so deploys need a manual model upload | [`.gitignore`](../.gitignore) | Low |
| Dependencies are floors, not pins — no lock file | [`requirements.txt`](../backend/requirements.txt) | Low |

**Next steps in order:** (1) source a non-leaked imaging dataset and re-run `audit_leakage.py` on
it; (2) wire or delete `/auth`; (3) set a real CORS origin list and an upload size cap;
(4) `pip freeze > requirements.lock.txt`.

---

## Reproducing every number in these docs

```bash
python ml/textual/evaluate.py           # 0.8403 acc / 0.8859 AUC / 0.7309 sens
python ml/ultrasound/audit_leakage.py   # leakage verdict + 88% width baseline
python ml/ultrasound/src/prepare_split.py
python ml/ultrasound/src/train.py       # ~25 min on CPU; result is leaked, see above
```

Verified on: Python 3.13.5, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.8.0, pandas 3.0.0,
Node 24.12.0.
