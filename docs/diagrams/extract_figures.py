"""Export result figures for the docs.

- ROC curve and confusion matrix: copied from the outputs stored in the
  final notebook, so they match the published run exactly (no data needed).
- Feature importance: drawn from the saved, elected XGBoost model.

Run from the repository root:
    .venv/Scripts/python docs/diagrams/extract_figures.py
"""
import base64
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "src" / "results_notebook" / "flight_delay_classification.ipynb"
MODEL = ROOT / "src" / "models" / "best_model_xgb.pkl"
OUT = ROOT / "docs" / "img"

# cell source marker -> output file name
FROM_NOTEBOOK = {
    "fpr_test, tpr_test, _ = roc_curve": "roc_curve.png",
    "ConfusionMatrixDisplay(confusion_matrix=conf_matrix": "confusion_matrix.png",
}

# readable names for the model's input columns
LABELS = {
    "Day_Of_Week": "Day of week", "Flight_Duration": "Flight duration", "Aicraft_age": "Aircraft age",
    "LATITUDE": "Departure latitude", "LONGITUDE": "Departure longitude", "tmin": "Min. temperature",
    "prcp": "Precipitation", "wspd": "Wind speed", "pres": "Air pressure", "Month": "Month",
    "Day": "Day of month", "Tail_Number_te": "Aircraft (tail number)*", "Dep_Airport_te": "Departure airport*",
    "DepTime_label_te": "Departure time slot*", "Arr_Airport_te": "Arrival airport*",
    "Distance_type_te": "Distance band*",
}


def export_notebook_images() -> None:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for marker, name in FROM_NOTEBOOK.items():
        cell = next(c for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"]))
        png = next(o["data"]["image/png"] for o in cell["outputs"] if "image/png" in o.get("data", {}))
        (OUT / name).write_bytes(base64.b64decode(png))
        print("wrote", name)


def plot_feature_importance() -> None:
    with MODEL.open("rb") as f:
        model = pickle.load(f)
    gain = model.get_booster().get_score(importance_type="gain")
    total = sum(gain.values())
    items = sorted(((LABELS.get(k, k), v / total) for k, v in gain.items()), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=150)
    names, values = zip(*items)
    colors = ["#0ea5e9" if n.endswith("*") else "#94a3b8" for n in names]
    ax.barh(names, values, color=colors)
    for y, v in enumerate(values):
        ax.text(v + 0.003, y, f"{v:.1%}", va="center", fontsize=8, color="#334155")
    ax.set_xlabel("Share of total gain")
    ax.set_title("Elected XGBoost: what drives the prediction", loc="left", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.01, "* target-encoded categorical feature", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "feature_importance_xgb.png", facecolor="white")
    print("wrote feature_importance_xgb.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export_notebook_images()
    plot_feature_importance()
