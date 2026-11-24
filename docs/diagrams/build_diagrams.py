"""Build the SVG diagrams used in README.md and docs/report.md.

Every number below is copied from the stored outputs of
src/results_notebook/flight_delay_classification.ipynb, so the diagrams
can be rebuilt without the 1.2 GB dataset. Standard library only.

Run from the repository root:
    .venv/Scripts/python docs/diagrams/build_diagrams.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "img"

# ---------------------------------------------------------------- palette
BG = "#0b1220"        # card background (opaque: reads well on light and dark GitHub themes)
PANEL = "#121c31"
PANEL_2 = "#17233d"
BORDER = "#26365a"
TEXT = "#e6edf7"
MUTED = "#9aa8c1"
FAINT = "#5d6b86"
SKY = "#38bdf8"       # data
VIOLET = "#a78bfa"    # preparation
AMBER = "#fbbf24"     # features
EMERALD = "#34d399"   # modeling / kept
ROSE = "#fb7185"      # dropped
FONT = "'Segoe UI', -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- primitives
def t(x, y, s, size=14, fill=TEXT, weight=400, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def lines(x, y, s, width, size=13, fill=MUTED, lh=1.38, weight=400):
    out = []
    for i, ln in enumerate(textwrap.wrap(s, width)):
        out.append(t(x, y + i * size * lh, ln, size, fill, weight))
    return "\n".join(out)


def rect(x, y, w, h, fill=PANEL, stroke=BORDER, rx=12, sw=1, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def arrow(x1, y1, x2, y2, color=FAINT, sw=2, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
            f'marker-end="url(#arrow)" {d}/>')


def svg(w, h, title, subtitle, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}" role="img" aria-label="{escape(title)}">
<title>{escape(title)}</title>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{FAINT}"/>
  </marker>
</defs>
{rect(0, 0, w, h, BG, BG, 20)}
{t(40, 54, title, 26, TEXT, 700)}
{t(40, 82, subtitle, 14, MUTED)}
{body}
</svg>
"""


def write(name, content):
    (OUT / name).write_text(content, encoding="utf-8", newline="\n")
    print("wrote", name)


# ---------------------------------------------------------------- 1. overview
def pipeline_overview():
    W, H = 1280, 790
    phases = [
        ("1", "Data integration", SKY, [
            ("US domestic flights 2023", "6,743,404 flights x 24 columns (schedule, airline, aircraft, delays)"),
            ("Daily weather per airport", "132,860 rows: temperature, rain, snow, wind, pressure"),
            ("Airport geolocation", "364 airports with latitude and longitude"),
            ("Left joins", "Flights + location on departure airport; + weather on airport and date"),
        ]),
        ("2", "Cleaning & sampling", VIOLET, [
            ("Only what is known in advance", "15 numeric + 8 categorical features; all delay-cause columns dropped"),
            ("Deduplicate", "853 duplicates removed; 37.9% of flights are delayed (>15 min)"),
            ("Sample and split", "25% random sample (1.69M), then 80/20 train/test split"),
            ("Outlier removal", "IsolationForest flags 5% of rows; class balance unchanged"),
        ]),
        ("3", "Feature engineering", AMBER, [
            ("K-fold target encoding", "8 categoricals (airline, aircraft, airports...) encoded out-of-fold"),
            ("Skew correction", "Box-Cox, cube root, log1p and sqrt picked per column"),
            ("Standard scaling", "Used for statistical tests and the linear model"),
            ("Feature selection", "ANOVA (p < 0.01) + correlation filter: 23 to 16 features"),
        ]),
        ("4", "Modeling & selection", EMERALD, [
            ("Class balancing", "Random undersampling to 50/50 (254,478 training rows)"),
            ("Baselines", "Random guess, Logistic Regression, RF, GB, XGBoost, LightGBM"),
            ("Randomized search", "20 candidates x 5-fold CV per model, scored on F1"),
            ("Hyperopt (TPE)", "50 trials each for XGBoost and LightGBM; XGBoost elected"),
        ]),
    ]
    col_w, gap, x0 = 280, 26, 40
    body = []
    for i, (num, name, color, steps) in enumerate(phases):
        x = x0 + i * (col_w + gap)
        body.append(rect(x, 112, col_w, 52, PANEL_2, color, 12, 1.5))
        body.append(f'<circle cx="{x + 28}" cy="138" r="14" fill="{color}"/>')
        body.append(t(x + 28, 143, num, 14, BG, 700, "middle"))
        body.append(t(x + 52, 144, name, 17, TEXT, 650))
        if i < len(phases) - 1:
            body.append(arrow(x + col_w + 3, 138, x + col_w + gap - 3, 138))
        y = 184
        for title, detail in steps:
            body.append(rect(x, y, col_w, 100))
            body.append(f'<rect x="{x}" y="{y + 14}" width="4" height="72" rx="2" fill="{color}"/>')
            body.append(t(x + 20, y + 30, title, 15, TEXT, 600))
            body.append(lines(x + 20, y + 54, detail, 37, 13))
            y += 112
    # results strip
    body.append(rect(40, 644, W - 80, 112, PANEL_2, EMERALD, 14, 1.5))
    body.append(t(64, 674, "RESULT ON 320,233 UNSEEN TEST FLIGHTS", 12, EMERALD, 700, extra='letter-spacing="1.5"'))
    kpis = [
        ("0.718", "ROC-AUC on the test set (train: 0.800)"),
        ("+66%", "recall on delayed flights vs. a base-rate guess (0.63 vs 0.38)"),
        ("+45%", "precision on delayed flights (0.55 vs 0.38)"),
        ("+55%", "F1 on delayed flights (0.59 vs 0.38)"),
    ]
    kw = (W - 80 - 48) / 4
    for i, (big, label) in enumerate(kpis):
        kx = 64 + i * kw
        body.append(t(kx, 716, big, 32, TEXT, 700))
        body.append(lines(kx + 112, 704, label, 26, 12.5, MUTED))
    write("pipeline_overview.svg", svg(W, H, "Flight delay prediction: end-to-end pipeline",
                                       "From three raw Kaggle tables to an XGBoost classifier that flags likely departure delays (>15 min) in advance",
                                       "\n".join(body)))


# ---------------------------------------------------------------- 2. data volume
def data_volume():
    W, H = 1180, 640
    rows = [
        ("Raw flights", "US_flights_2023.csv", 6_743_404, SKY, None),
        ("Deduplicated", "853 duplicate rows removed", 6_742_551, VIOLET, None),
        ("25% random sample", "for computational feasibility", 1_685_637, VIOLET, None),
        ("Train split (80%)", "test split: 337,128 rows", 1_348_509, VIOLET, None),
        ("Train without outliers", "IsolationForest, 5% removed; test: 320,233", 1_281_083, VIOLET, None),
        ("Training subsample", "25% of the train split size", 337_127, AMBER, None),
        ("Balanced training set", "random undersampling: 127,239 per class", 254_478, EMERALD, None),
    ]
    import math
    label_w, bar_x, bar_max = 330, 380, 620
    top = max(r[2] for r in rows)
    body = []
    y = 124
    for name, sub, n, color, _ in rows:
        w = max(8, bar_max * math.sqrt(n / top))
        body.append(t(40, y + 20, name, 16, TEXT, 600))
        body.append(t(40, y + 42, sub, 12.5, MUTED))
        body.append(rect(bar_x, y + 4, bar_max, 40, PANEL, BORDER, 8))
        body.append(rect(bar_x, y + 4, w, 40, color, color, 8, 0, 'fill-opacity="0.9"'))
        body.append(t(bar_x + w + 14, y + 30, f"{n:,}", 16, TEXT, 700))
        y += 66
    body.append(t(40, H - 30, "Bar length is proportional to the square root of the row count, so the smaller sets stay readable.", 12, FAINT))
    write("data_volume_funnel.svg", svg(W, H, "How many flights reach the model",
                                        "Row counts at each preparation step (training side unless noted)", "\n".join(body)))


# ---------------------------------------------------------------- 3. target encoding
def target_encoding():
    W, H = 1180, 620
    body = []
    body.append(t(40, 130, "Training set split into 5 stratified folds", 16, TEXT, 600))
    body.append(lines(40, 154, "In each round, category means are learned on 4 folds and written into the held-out fold, so rows are encoded from the other folds' statistics.", 70, 13))
    bx, by, bw, bh, g = 40, 200, 92, 46, 8
    for r in range(5):
        y = by + r * (bh + 14)
        body.append(t(bx, y + 29, f"Round {r + 1}", 13, MUTED, 600))
        for c in range(5):
            x = bx + 84 + c * (bw + g)
            if c == r:
                body.append(rect(x, y, bw, bh, AMBER, AMBER, 8, 1, 'fill-opacity="0.92"'))
                body.append(t(x + bw / 2, y + 28, "encode", 13, BG, 700, "middle"))
            else:
                body.append(rect(x, y, bw, bh, PANEL_2, SKY, 8, 1))
                body.append(t(x + bw / 2, y + 28, "learn means", 12, SKY, 600, "middle"))
    # legend
    ly = by + 5 * (bh + 14) + 18
    body.append(rect(bx + 84, ly, 14, 14, PANEL_2, SKY, 3))
    body.append(t(bx + 106, ly + 12, "folds used to compute category means", 12.5, MUTED))
    body.append(rect(bx + 380, ly, 14, 14, AMBER, AMBER, 3))
    body.append(t(bx + 402, ly + 12, "fold receiving the encoded values", 12.5, MUTED))
    # right column
    rx = 680
    body.append(rect(rx, 120, 460, 210, PANEL, AMBER, 14, 1.5))
    body.append(t(rx + 24, 154, "Smoothed category mean", 16, TEXT, 650))
    body.append(t(rx + 24, 198, "enc(c) = (mean_c * n_c + global * a) / (n_c + a)", 14, AMBER, 600,
                  extra='font-family="Consolas, Menlo, monospace"'))
    body.append(lines(rx + 24, 232, "mean_c and n_c = delay rate and flight count of category c, global = overall delay rate, a = 0.3. Rare categories (for example a single aircraft tail number) are pulled toward the overall delay rate instead of memorising a few labels.", 56, 13))
    body.append(rect(rx, 350, 460, 150, PANEL, SKY, 14, 1.5))
    body.append(t(rx + 24, 384, "Test set", 16, TEXT, 650))
    body.append(lines(rx + 24, 410, "Mapped with the category means learned on the training folds. Categories never seen in training get the global mean (only 18 rows in the whole test split).", 56, 13))
    body.append(lines(40, H - 52, "Applied to 8 columns: Airline, Tail_Number, Dep_Airport, DepTime_label, Arr_Airport, Distance_type, Manufacturer, Model. It turns high-cardinality IDs (thousands of aircraft) into one informative number each, without one-hot explosion.", 150, 12.5, FAINT))
    write("target_encoding_kfold.svg", svg(W, H, "Out-of-fold (K-fold) target encoding",
                                           "How categorical columns become numbers while keeping each row's own label out of its encoding", "\n".join(body)))


# ---------------------------------------------------------------- 4. feature selection
def feature_selection():
    W, H = 1180, 480
    feats = [
        ("Day_Of_Week", "keep"), ("Flight_Duration", "keep"), ("Aicraft_age", "keep"), ("LATITUDE", "keep"),
        ("LONGITUDE", "keep"), ("tavg", "corr"), ("tmin", "keep"), ("tmax", "corr"), ("prcp", "keep"),
        ("snow", "anova"), ("wdir", "anova"), ("wspd", "keep"), ("pres", "keep"), ("Month", "keep"),
        ("Day", "keep"), ("Airline_te", "corr"), ("Tail_Number_te", "keep"), ("Dep_Airport_te", "keep"),
        ("DepTime_label_te", "keep"), ("Arr_Airport_te", "keep"), ("Distance_type_te", "keep"),
        ("Manufacturer_te", "corr"), ("Model_te", "corr"),
    ]
    style = {"keep": (EMERALD, "kept"), "anova": (AMBER, "dropped by ANOVA"), "corr": (ROSE, "dropped by correlation filter")}
    body = []
    # counters
    stages = [("23", "candidate features", "15 numeric + 8 target-encoded", SKY),
              ("21", "after ANOVA F-test", "p-value < 0.01 against the target", AMBER),
              ("16", "after correlation filter", "|r| > 0.7: keep the one closer to the target", EMERALD)]
    sx = 40
    for i, (n, name, sub, color) in enumerate(stages):
        body.append(rect(sx, 112, 340, 92, PANEL, color, 14, 1.5))
        body.append(t(sx + 24, 172, n, 44, color, 700))
        body.append(t(sx + 100, 150, name, 16, TEXT, 650))
        body.append(t(sx + 100, 174, sub, 12.5, MUTED))
        if i < 2:
            body.append(arrow(sx + 343, 158, sx + 377, 158))
        sx += 380
    # chips
    x, y, chip_h = 40, 240, 38
    for name, state in feats:
        color, _ = style[state]
        w = 22 + len(name) * 8.2
        if x + w > W - 40:
            x, y = 40, y + chip_h + 12
        if state == "keep":
            body.append(rect(x, y, w, chip_h, PANEL_2, color, 19, 1.5))
            body.append(t(x + w / 2, y + 24, name, 13, TEXT, 600, "middle"))
        else:
            body.append(rect(x, y, w, chip_h, BG, color, 19, 1.2, 'stroke-dasharray="5 4"'))
            body.append(t(x + w / 2, y + 24, name, 13, color, 500, "middle", 'text-decoration="line-through"'))
        x += w + 10
    # legend
    ly = y + chip_h + 36
    lx = 40
    for state in ("keep", "anova", "corr"):
        color, label = style[state]
        body.append(f'<circle cx="{lx + 7}" cy="{ly - 4}" r="7" fill="{color}"/>')
        body.append(t(lx + 22, ly + 1, label, 13, MUTED))
        lx += 60 + len(label) * 7.5
    body.append(lines(40, ly + 40, "Correlated pairs: the three temperatures collapse to tmin; airline, manufacturer and model are largely implied by the individual aircraft, so Tail_Number_te (more correlated with the target) is kept.", 150, 12.5, FAINT))
    write("feature_selection_funnel.svg", svg(W, H, "Feature selection: 23 to 16",
                                              "Statistical relevance first, then redundancy removal", "\n".join(body)))


# ---------------------------------------------------------------- 5. model selection
def model_selection():
    W = 1180
    stages = [
        ("Random guess", "predicts by class proportions", [("Base rate", 0.38)], FAINT),
        ("Logistic Regression", "scaled features, undersampled", [("LogReg", 0.57)], VIOLET),
        ("Default ensembles", "out of the box, undersampled", [("RandomForest", 0.57), ("GradientBoost", 0.58), ("XGBoost", 0.59), ("LightGBM", 0.58)], SKY),
        ("RandomizedSearchCV", "20 candidates x 5-fold, F1", [("RandomForest", 0.58), ("XGBoost", 0.59), ("LightGBM", 0.59)], AMBER),
        ("Hyperopt TPE", "50 trials x stratified 5-fold, F1", [("XGBoost", 0.59), ("LightGBM", 0.59)], EMERALD),
    ]
    body = []
    scale_x, scale_w, vmax = 470, 520, 0.7
    y = 116
    body.append(t(scale_x, y, "F1-score on delayed flights (class 1), test set", 12.5, MUTED, 600))
    base_x = scale_x + scale_w * 0.38 / vmax
    y = 136
    top_y = y
    for name, sub, models, color in stages:
        h = max(72, 22 + len(models) * 30)
        body.append(rect(40, y, W - 80, h, PANEL, BORDER, 12))
        body.append(f'<rect x="40" y="{y + 10}" width="4" height="{h - 20}" rx="2" fill="{color}"/>')
        body.append(t(64, y + 30, name, 16, TEXT, 650))
        body.append(t(64, y + 52, sub, 12.5, MUTED))
        for j, (m, v) in enumerate(models):
            my = y + 14 + j * 30
            body.append(t(scale_x - 14, my + 15, m, 13, MUTED, 500, "end"))
            body.append(rect(scale_x, my + 2, scale_w * v / vmax, 18, color, color, 5, 0, 'fill-opacity="0.85"'))
            body.append(t(scale_x + scale_w * v / vmax + 10, my + 16, f"{v:.2f}", 13, TEXT, 700))
        y += h + 12
    body.append(f'<line x1="{base_x}" y1="{top_y - 4}" x2="{base_x}" y2="{y - 8}" stroke="{ROSE}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    body.append(t(base_x + 6, y + 6, "base rate 0.38", 12, ROSE, 600))
    # elected card
    ey = y + 24
    body.append(rect(40, ey, W - 80, 118, PANEL_2, EMERALD, 14, 1.5))
    body.append(t(64, ey + 32, "Elected: XGBoost (Hyperopt)", 18, EMERALD, 700))
    body.append(t(64, ey + 56, "275 trees, max depth 8, learning rate 0.064, subsample 0.85, colsample 0.8", 13, MUTED))
    body.append(lines(64, ey + 82, "Chosen on cross-validated F1 (0.648 vs 0.647 for LightGBM). Test: accuracy 0.67, class-1 precision 0.55, recall 0.63, ROC-AUC 0.718.", 120, 13, TEXT))
    H = ey + 118 + 32
    write("model_selection.svg", svg(W, H, "Model selection",
                                     "Each stage is compared on the same 320,233 held-out flights", "\n".join(body)))


# ---------------------------------------------------------------- 6. experiment timeline
def experiment_timeline():
    W, H = 1180, 900
    items = [
        ("01", "Resampling strategies", "NearMiss vs random undersampling; also tried an MLP and CatBoost.", "random undersampling kept", EMERALD),
        ("02", "Arrival-delay regression", "Predict minutes of arrival delay. R2 reached 0.97, but only because departure delay was an input.", "objective dropped", ROSE),
        ("03", "One-hot + target encoding", "Mixed encodings for categoricals, NearMiss balancing, LightGBM tuned with Hyperopt.", "target encoding only", EMERALD),
        ("04", "Last-aircraft delay feature", "Delay of the aircraft's previous flight as an input, plus decision-threshold tuning.", "dropped: not known in advance", ROSE),
        ("05", "Outliers and skew", "IsolationForest outlier removal and automatic skew transforms.", "kept", EMERALD),
        ("06", "Pre-final pipeline", "Full final pipeline on a 10% sample, still using the last-aircraft delay.", "refined into the final", AMBER),
        ("07", "Regression port", "Final notebook partly converted to regression; never re-run.", "abandoned", ROSE),
    ]
    body = []
    lx = 92
    y = 130
    body.append(f'<line x1="{lx}" y1="{y}" x2="{lx}" y2="{y + len(items) * 86 + 40}" stroke="{BORDER}" stroke-width="3"/>')
    for num, name, desc, verdict, color in items:
        body.append(f'<circle cx="{lx}" cy="{y + 24}" r="20" fill="{PANEL_2}" stroke="{color}" stroke-width="2"/>')
        body.append(t(lx, y + 29, num, 13, TEXT, 700, "middle"))
        body.append(rect(lx + 40, y, 1000, 66, PANEL, BORDER, 12))
        body.append(t(lx + 60, y + 27, name, 15.5, TEXT, 650))
        body.append(t(lx + 60, y + 50, desc, 12.5, MUTED))
        vw = 26 + len(verdict) * 7.2
        body.append(rect(lx + 1020 - vw, y + 13, vw, 26, BG, color, 13, 1.2))
        body.append(t(lx + 1020 - vw / 2, y + 31, verdict, 12, color, 600, "middle"))
        y += 86
    body.append(f'<circle cx="{lx}" cy="{y + 34}" r="24" fill="{EMERALD}"/>')
    body.append(t(lx, y + 41, "★", 20, BG, 700, "middle"))
    body.append(rect(lx + 40, y, 1000, 80, PANEL_2, EMERALD, 12, 1.5))
    body.append(t(lx + 60, y + 30, "Final: flight_delay_classification.ipynb", 16, EMERALD, 700))
    body.append(t(lx + 60, y + 56, "Only pre-flight information, outlier removal, K-fold target encoding, ANOVA + correlation selection, Hyperopt-tuned XGBoost.", 12.5, TEXT))
    body.append(t(40, H - 30, "Also tried along the way: HDBSCAN clustering to train one model per flight group (no gain for a high compute cost).", 12.5, FAINT))
    write("experiment_timeline.svg", svg(W, H, "Experiment log",
                                         "Archived notebooks in src/notebooks/archive and what each one taught", "\n".join(body)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline_overview()
    data_volume()
    target_encoding()
    feature_selection()
    model_selection()
    experiment_timeline()
