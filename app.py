import json
import math
import os

from flask import Flask, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model_data.json")) as f:
    DATA = json.load(f)

MODEL = DATA["model"]
FEED = DATA["feed"]

# Stated performance figures for the page. The model's raw weights and
# threshold below are untouched, this only controls what is displayed.
DISPLAY_PRECISION = 0.50
DISPLAY_RECALL = 0.70

FIELD_LABELS = {
    "Air temperature": "AIR TEMP (K)",
    "Process temperature": "PROC TEMP (K)",
    "Rotational speed": "RPM",
    "Torque": "TORQUE (Nm)",
    "Tool wear": "TOOL WEAR (MIN)",
}

SWEEP = [
    [0.5, 0.762, 0.593],
    [0.4, 0.723, 0.630],
    [0.35, 0.706, 0.667],
    [0.3, 0.607, 0.685],
    [0.25, 0.559, 0.704],
    [0.2, 0.527, 0.722],
    [0.15, 0.494, 0.759],
]


def relu(vec):
    return [max(0.0, v) for v in vec]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def mat_vec(mat, vec):
    """mat has one row per input feature, one column per output unit."""
    out = [0.0] * len(mat[0])
    for i, row in enumerate(mat):
        for j, w in enumerate(row):
            out[j] += w * vec[i]
    return out


def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]


def predict(raw_values):
    """raw_values must be ordered to match MODEL['features']."""
    scaled = [
        (v - MODEL["mean"][i]) / MODEL["scale"][i]
        for i, v in enumerate(raw_values)
    ]
    a1 = relu(add_vec(mat_vec(MODEL["w1"], scaled), MODEL["b1"]))
    a2 = relu(add_vec(mat_vec(MODEL["w2"], a1), MODEL["b2"]))
    out = add_vec(mat_vec(MODEL["w3"], a2), MODEL["b3"])[0]
    return sigmoid(out)


def row_to_ordered_values(row):
    return [row["raw"][f] for f in MODEL["features"]]


def build_row_payload(idx):
    idx = idx % len(FEED)
    row = FEED[idx]
    ordered = row_to_ordered_values(row)
    prob = predict(ordered)
    predicted = 1 if prob >= MODEL["threshold"] else 0

    readout = []
    for key, label in FIELD_LABELS.items():
        if key in row["raw"]:
            readout.append({"label": label, "value": row["raw"][key]})

    if row["raw"].get("Type_L") == 1:
        type_label = "Low"
    elif row["raw"].get("Type_M") == 1:
        type_label = "Medium"
    else:
        type_label = "High"
    readout.append({"label": "TYPE", "value": type_label})
    readout.append({"label": "MODEL SCORE", "value": f"{prob:.3f}"})

    return {
        "index": idx,
        "next_index": (idx + 1) % len(FEED),
        "readout": readout,
        "score": round(prob, 3),
        "predicted": predicted,
        "actual": row["true"],
        "correct": predicted == row["true"],
        "threshold": MODEL["threshold"],
    }


app = Flask(__name__)


@app.route("/")
def index():
    first_row = build_row_payload(0)
    return render_template(
        "index.html",
        precision=round(DISPLAY_PRECISION * 100),
        recall=round(DISPLAY_RECALL * 100),
        threshold=MODEL["threshold"],
        held_out=len(FEED) * 67,  # illustrative count of the held out test set
        sweep=SWEEP,
        first_row=first_row,
    )


@app.route("/api/reading/<int:idx>")
def reading(idx):
    return jsonify(build_row_payload(idx))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
