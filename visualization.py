"""
Visualization Module
=====================
Generates all charts and plots for the Mental Health Alert System:
  1. Emotion / Category distribution (bar + pie)
  2. Risk level breakdown
  3. Model accuracy comparison
  4. Confusion matrices (heatmaps)
  5. TF-IDF word clouds
  6. Training data class distribution
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Style Config ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "axes.titlecolor":  "#f0f6fc",
    "text.color":       "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "grid.color":       "#21262d",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})

PALETTE = {
    "Normal Conversation":        "#2ea043",
    "Stress / Anxiety":           "#d29922",
    "Depression / Sadness":       "#a371f7",
    "Suicide Risk":               "#f85149",
    "Cyberbullying":              "#ff7b72",
    "Violence / Threats":         "#f0883e",
    "Trust / Relationship Issues":"#58a6ff",
}

RISK_COLORS = {"low": "#2ea043", "medium": "#d29922", "high": "#f85149"}
MODEL_COLORS = {"Logistic Regression": "#2ea043", "Support Vector Machine": "#a371f7", "Random Forest": "#d29922"}

LABEL_ORDER = [
    "Normal Conversation", "Stress / Anxiety", "Depression / Sadness",
    "Suicide Risk", "Cyberbullying", "Violence / Threats", "Trust / Relationship Issues",
]


def _save(fig, path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    full = os.path.join(output_dir, path)
    fig.savefig(full, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return full


# ── 1. Category Distribution ─────────────────────────────────
def plot_category_distribution(df: pd.DataFrame, output_dir="outputs") -> str:
    counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    colors = [PALETTE.get(l, "#888") for l in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d1117")
    fig.suptitle("Emotion & Risk Category Distribution", fontsize=15, fontweight="bold",
                 color="#f0f6fc", y=1.01)

    # Bar chart
    ax = axes[0]
    bars = ax.barh(counts.index, counts.values, color=colors, height=0.65, alpha=0.9)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f" {val}", va="center", fontsize=10, color="#c9d1d9")
    ax.set_xlabel("Number of Samples", fontsize=11)
    ax.set_title("Sample Count per Category", fontsize=12, color="#f0f6fc")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.invert_yaxis()

    # Pie chart
    ax2 = axes[1]
    wedge_props = {"linewidth": 2, "edgecolor": "#0d1117"}
    ax2.pie(counts.values, labels=None, colors=colors, autopct="%1.1f%%",
            pctdistance=0.82, startangle=140, wedgeprops=wedge_props,
            textprops={"color": "#c9d1d9", "fontsize": 9})
    patches = [mpatches.Patch(color=colors[i], label=l) for i, l in enumerate(counts.index)]
    ax2.legend(handles=patches, loc="upper left", bbox_to_anchor=(-0.15, 1.0),
               fontsize=8, framealpha=0.2)
    ax2.set_title("Category Proportion", fontsize=12, color="#f0f6fc")

    plt.tight_layout()
    return _save(fig, "category_distribution.png", output_dir)


# ── 2. Risk Level Breakdown ──────────────────────────────────
def plot_risk_levels(df: pd.DataFrame, output_dir="outputs") -> str:
    RISK_MAP = {
        "Normal Conversation":        "low",
        "Stress / Anxiety":           "medium",
        "Depression / Sadness":       "medium",
        "Suicide Risk":               "high",
        "Cyberbullying":              "high",
        "Violence / Threats":         "high",
        "Trust / Relationship Issues":"medium",
    }
    df = df.copy()
    df["risk_level"] = df["label"].map(RISK_MAP)
    risk_counts = df["risk_level"].value_counts().reindex(["low","medium","high"], fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    colors = [RISK_COLORS[r] for r in risk_counts.index]
    bars = ax.bar(["🟢 Low Risk\n(Normal)", "🟡 Medium Risk\n(Stress/Trust)", "🔴 High Risk\n(Crisis)"],
                  risk_counts.values, color=colors, width=0.5, alpha=0.9)
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=13, fontweight="bold", color="#f0f6fc")
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_title("Risk Level Distribution", fontsize=14, fontweight="bold", color="#f0f6fc")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right","bottom"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "risk_level_distribution.png", output_dir)


# ── 3. Model Accuracy Comparison ─────────────────────────────
def plot_model_comparison(results: dict, output_dir="outputs") -> str:
    models   = list(results.keys())
    metrics  = ["accuracy", "precision", "recall", "f1_score"]
    x        = np.arange(len(metrics))
    width    = 0.22
    colors   = [MODEL_COLORS.get(m, "#888") for m in models]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0d1117")
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [results[model][m] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=color, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#f0f6fc")

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ML Model Performance Comparison", fontsize=14, fontweight="bold", color="#f0f6fc")
    ax.legend(fontsize=10, framealpha=0.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return _save(fig, "model_comparison.png", output_dir)


# ── 4. Confusion Matrix ───────────────────────────────────────
def plot_confusion_matrix(cm_data: list, model_name: str,
                          labels: list = LABEL_ORDER, output_dir="outputs") -> str:
    cm = np.array(cm_data)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)

    short_labels = [l.replace(" / ", "/").replace(" Issues","").replace(" Conversation","") for l in labels]

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0d1117")
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035)
    cbar.ax.tick_params(labelcolor="#c9d1d9")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "#0d1117" if cm_norm[i, j] > 0.5 else "#c9d1d9"
            ax.text(j, i, f"{val}", ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", color="#f0f6fc")
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    return _save(fig, fname, output_dir)


# ── 5. Per-Class F1 Scores ───────────────────────────────────
def plot_per_class_f1(results: dict, output_dir="outputs") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")
    fig.suptitle("Per-Class F1 Scores by Model", fontsize=14, fontweight="bold", color="#f0f6fc")

    for ax, (model_name, res) in zip(axes, results.items()):
        report = res.get("classification_report", {})
        f1s = []
        for label in LABEL_ORDER:
            f1s.append(report.get(label, {}).get("f1-score", 0))

        short = [l.split("/")[0].strip()[:18] for l in LABEL_ORDER]
        colors = [PALETTE.get(l, "#888") for l in LABEL_ORDER]
        bars = ax.barh(short, f1s, color=colors, alpha=0.85, height=0.6)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=8.5, color="#c9d1d9")
        ax.set_xlim(0, 1.15)
        ax.set_title(model_name, fontsize=11, color=MODEL_COLORS.get(model_name,"#fff"))
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        ax.invert_yaxis()

    plt.tight_layout()
    return _save(fig, "per_class_f1.png", output_dir)


# ── 6. Full Dashboard ────────────────────────────────────────
def plot_full_dashboard(df: pd.DataFrame, results: dict, output_dir="outputs") -> str:
    """Single comprehensive dashboard figure."""
    RISK_MAP = {
        "Normal Conversation":        "low",
        "Stress / Anxiety":           "medium",
        "Depression / Sadness":       "medium",
        "Suicide Risk":               "high",
        "Cyberbullying":              "high",
        "Violence / Threats":         "high",
        "Trust / Relationship Issues":"medium",
    }

    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    gs  = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    fig.text(0.5, 0.97, "🛡️  SENTINEL AI — Mental Health Early Alert System Dashboard",
             ha="center", fontsize=16, fontweight="bold", color="#f0f6fc")
    fig.text(0.5, 0.945, "NLP + Machine Learning Risk Analysis",
             ha="center", fontsize=11, color="#8b949e")

    # 1. Category bar
    ax1 = fig.add_subplot(gs[0, :2])
    counts = df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    colors = [PALETTE.get(l, "#888") for l in counts.index]
    ax1.barh(counts.index, counts.values, color=colors, height=0.65, alpha=0.9)
    ax1.set_title("Category Distribution", color="#f0f6fc", fontsize=12)
    ax1.set_xlabel("Samples")
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.invert_yaxis()

    # 2. Risk pie
    ax2 = fig.add_subplot(gs[0, 2])
    df2 = df.copy()
    df2["risk"] = df2["label"].map(RISK_MAP)
    rc = df2["risk"].value_counts().reindex(["low","medium","high"], fill_value=0)
    ax2.pie(rc.values, labels=["🟢 Low","🟡 Medium","🔴 High"],
            colors=[RISK_COLORS[r] for r in rc.index],
            autopct="%1.1f%%", startangle=90,
            textprops={"color":"#c9d1d9","fontsize":9},
            wedgeprops={"linewidth":2,"edgecolor":"#0d1117"})
    ax2.set_title("Risk Level Split", color="#f0f6fc", fontsize=12)

    # 3. Model comparison
    ax3 = fig.add_subplot(gs[0, 3])
    model_names = list(results.keys())
    f1s = [results[m]["f1_score"] for m in model_names]
    bars = ax3.bar(["LR","SVM","RF"], f1s,
                   color=[MODEL_COLORS.get(m, "#888") for m in model_names],
                   alpha=0.9, width=0.5)
    for b, v in zip(bars, f1s):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{v:.3f}", ha="center", fontsize=9, color="#f0f6fc")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Model F1 Scores", color="#f0f6fc", fontsize=12)
    ax3.set_ylabel("F1 Score")
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines[["top","right"]].set_visible(False)

    # 4. Three confusion matrices (small)
    for col, (mname, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, col + (1 if col == 2 else col)])
        cm = np.array(res["confusion_matrix"])
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)
        short = [l[:8] for l in LABEL_ORDER]
        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(short)))
        ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
        ax.set_title(mname.replace(" ", "\n"), color=MODEL_COLORS.get(mname,"#fff"), fontsize=9)

    # 5. Metrics table
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis("off")
    table_data = [[m[:6], f"{results[m]['accuracy']:.3f}", f"{results[m]['f1_score']:.3f}"]
                  for m in model_names]
    headers = ["Model", "Acc.", "F1"]
    t = ax5.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor("#21262d" if r == 0 else "#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#f0f6fc")
    ax5.set_title("Performance\nSummary", color="#f0f6fc", fontsize=10)

    # 6. NLP pipeline visual
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    pipeline_steps = [
        "📥 Raw Text", "→  Lowercase", "→  Remove URLs", "→  Clean Chars",
        "→  Tokenize", "→  Remove\nStopwords", "→  Lemmatize", "→  TF-IDF\nVectors",
        "→  ML Model", "→  Risk\nClassification", "→  Alert\nSystem"
    ]
    for i, step in enumerate(pipeline_steps):
        x = 0.01 + i * 0.092
        ax6.text(x, 0.5, step, transform=ax6.transAxes,
                 ha="center", va="center", fontsize=8.5,
                 color="#f0f6fc" if "→" not in step else "#8b949e",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#21262d" if "→" not in step else "none",
                           ec="#30363d" if "→" not in step else "none", alpha=0.9))
    ax6.set_title("NLP + ML Processing Pipeline", color="#f0f6fc", fontsize=11, pad=12)

    return _save(fig, "full_dashboard.png", output_dir)


if __name__ == "__main__":
    # Quick test with dummy data
    df = pd.DataFrame({
        "label": LABEL_ORDER * 28 + LABEL_ORDER[:2],
        "text": ["sample text"] * 198,
    })
    mock_results = {
        m: {"accuracy": 0.85, "precision": 0.84, "recall": 0.85, "f1_score": 0.84,
            "confusion_matrix": np.eye(7, dtype=int).tolist(),
            "classification_report": {l: {"f1-score": 0.85} for l in LABEL_ORDER}}
        for m in ["Logistic Regression", "Support Vector Machine", "Random Forest"]
    }
    os.makedirs("outputs", exist_ok=True)
    p = plot_full_dashboard(df, mock_results, "outputs")
    print(f"Dashboard saved: {p}")
