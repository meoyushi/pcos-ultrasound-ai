# 02 — The Two Models

Two independent models, no shared code. Both are reproducible from checked-in scripts.

| | Clinical (textual) | Imaging (ultrasound) |
|---|---|---|
| Library | scikit-learn | TensorFlow 2.21 / Keras 3.15 |
| Model | `RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced")` | EfficientNetB0 + GAP + Dense(64) + Dense(1, sigmoid) |
| Data | 541-patient spreadsheet | 1,921 unique ultrasound images |
| Trained | at API startup, in-process | offline, `ml/ultrasound/src/train.py` |
| Artifact | none — refit each boot | `backend/models/pcos_efficientnet.h5` (34 MB) |
| **Validated?** | **Yes — 0.84 acc / 0.89 AUC** | **No — dataset is leaked, see [03-findings](03-findings.md)** |

---

## Clinical model — Random Forest

### Data

[`backend/data/PCOS_data_without_infertility.xlsx`](../backend/data/PCOS_data_without_infertility.xlsx),
sheet `Full_new`: **541 rows x 45 columns**, target `PCOS (Y/N)` with **364 negative / 177 positive**
(32.7% prevalence). The public Kaggle PCOS dataset from Kerala fertility clinics; no ingestion
script, the file is simply checked in.

Quirks the code handles: header whitespace (`' Age (yrs)'`, `'Height(Cm) '`), `AMH(ng/mL)` typed as
object, `BMI` null in 299 of 541 rows, an empty `Unnamed: 44` column.

### Features — the important design decision

The model trains on **exactly the 13 fields the form collects**, plus BMI derived from weight and
height ([`FORM_FEATURES`](../backend/app/services/textual_service.py)):

| Group | Fields |
|---|---|
| Demographics | Age, Weight, Height, BMI *(derived)*, Blood Group |
| Menstrual | Cycle regularity, Cycle length |
| Symptoms | Weight gain, Hair growth, Skin darkening, Hair loss, Pimples |
| Lifestyle | Fast food, Regular exercise |

It previously trained on all 41 spreadsheet columns and median-filled the 28 the form does not
collect. Those 28 included `Follicle No. (L)` and `Follicle No. (R)` — antral follicle counts read
off an ultrasound, and the two strongest features in the data by a wide margin. Pinning them to a
constant meant the forest's dominant split always routed toward "Normal":

| Setup | Accuracy | ROC-AUC | **Sensitivity** |
|---|---|---|---|
| Train on 41, serve 13 + medians *(old)* | 0.747 | 0.874 | **0.259** |
| Train on 13, serve 13, unbounded trees | 0.820 | 0.881 | 0.646 |
| **Train on 13, serve 13, `max_depth=6`** *(current)* | **0.840** | **0.886** | **0.731** |
| Majority-class baseline | 0.673 | — | — |

The old pipeline missed roughly **three of every four PCOS cases**. Fixing the train/serve mismatch
nearly tripled sensitivity. Depth was capped because unbounded trees hit 1.000 accuracy on their own
training data.

### Preprocessing

In `load_dataset()`, shared with the evaluation script so both measure the same thing:

1. Drop `Sl. No`, `Patient File No.`, `Unnamed: 44`
2. Strip header whitespace
3. Recompute `BMI = kg / (m^2)` for every row
4. `pd.to_numeric(errors="coerce")` on object columns
5. Per-column median imputation
6. Select `FORM_FEATURES`

No scaling (correct for trees), no encoding (values are already numeric-coded).

### Validation

```bash
python ml/textual/evaluate.py     # writes ml/textual/metrics.json
```

RepeatedStratifiedKFold, 5 splits x 5 repeats (25 fits):

| Metric | Mean | Std |
|---|---|---|
| Accuracy | **0.8403** | 0.0316 |
| ROC-AUC | **0.8859** | 0.0318 |
| Sensitivity (recall) | 0.7309 | 0.0669 |
| Precision | 0.7707 | 0.0557 |
| F1 | 0.7489 | 0.0527 |

Feature importance, top 5: Skin darkening 0.163, Hair growth 0.128, Weight gain 0.116,
Cycle length 0.087, Cycle regularity 0.077. Sensible — hyperandrogenism and cycle irregularity are
two of the three Rotterdam criteria.

**Caveat to state aloud in an interview:** this is cross-validation on a single 541-patient,
single-source dataset. There is no external validation cohort.

### Inference

`predict()` requires every field. Missing or non-numeric values raise `ValueError` -> HTTP 400;
zero/negative height or weight is rejected. Nothing is silently median-filled any more — that
substitution is exactly what broke the old model.

---

## Imaging model — EfficientNetB0

### Data and split

3,846 image files under `data/`, which collapse to **1,921 unique images** by MD5 (1,925 duplicates).
`data/train/` and `data/test/` as shipped were **byte-identical copies of each other**, and the old
training script used `data/test/` as `validation_data` — so every metric it printed, and every
`EarlyStopping` decision it made, was on training data.

```bash
python ml/ultrasound/src/prepare_split.py
```

Hashes every image, collapses duplicates, asserts no image appears under two labels, and writes a
stratified **70/15/15** split to `ml/ultrasound/splits/*.csv` — index files, not copied pixels, so
the 140 MB is not duplicated:

```
train n=1344  {Normal: 799, PCOS: 545}
val   n= 288  {Normal: 171, PCOS: 117}
test  n= 289  {Normal: 172, PCOS: 117}
Verified: train, val and test are pairwise disjoint by image content.
```

### Architecture and training

```python
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)
```

Two phases ([`train.py`](../ml/ultrasound/src/train.py)):

| Phase | Backbone | Epochs | LR | Trainable params |
|---|---|---|---|---|
| 1 | frozen | 8 | 1e-3 | 82,049 |
| 2 | top 38 layers unfrozen, BatchNorm kept frozen | 12 | 1e-5 | 2,119,569 |

Loss `binary_crossentropy`, Adam, class weights `{Normal: 0.84, PCOS: 1.23}`, augmentation
(rotation 20°, shift/shear 0.15, zoom 0.2, horizontal flip). `EarlyStopping(restore_best_weights)`
and `ReduceLROnPlateau` both monitor **val**, which is now genuinely held out.

### Preprocessing — the bug that mattered

Keras' EfficientNetB0 normalises **inside the graph** and expects raw **0-255** input. Both the
training generator (`rescale=1./255`) and the serving preprocessor (`/ 255.0`) divided first,
feeding the pretrained filters values ~255x smaller than the statistics they were trained on. Both
are now removed and the two files must stay in agreement.

Evidence this was real, measured on the original `.h5` before it was overwritten
(backup at `/tmp/pcos_efficientnet_ORIGINAL.h5`):

| Input path | Output range | Std |
|---|---|---|
| As trained (`/255`) | 0.4218 - 0.4267 | 0.0009 |

The old model emitted a near-constant **~0.423** for every image. The serving code overrode the
network whenever its output fell inside `(0.1, 0.9)` — so the pixel-variance heuristic fired on
**100% of predictions**. The deployed classifier contained zero EfficientNetB0 contribution. That
override is now off by default (`ENABLE_PIXEL_STD_FALLBACK = False`) and every current response
carries `used_pixel_std_fallback: false`.

### Results — and why you must not quote them

```bash
python ml/ultrasound/src/train.py     # writes ml/ultrasound/metrics.json
```

Held-out test (289 unseen images): accuracy **1.0000**, ROC-AUC **1.0000**, sensitivity 1.0000,
specificity 1.0000.

**This is a symptom, not a success.** `val_accuracy` reached 1.0000 after one epoch with the
backbone still frozen. The dataset is leaked — the two class folders are separable by image
resolution alone. Read [03-findings.md](03-findings.md) before repeating any of these numbers.
`ml/ultrasound/metrics.json` carries a `METRICS_ARE_NOT_VALID` field for the same reason.
