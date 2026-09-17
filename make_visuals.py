"""Generate the README charts and animations from real training artefacts.

Every number plotted here comes from a training log or a measured evaluation in
this repository. Nothing is illustrative or synthetic.

    python make_visuals.py

Writes PNGs and GIFs into docs/assets/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (light surface). See dataviz reference palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 160,
})


def style(ax, title, subtitle=None, xlabel=None, ylabel=None):
    # Offsets in points, not axes fractions, so spacing holds at any figure height.
    ax.set_title(title, color=INK, fontsize=14, fontweight="bold", loc="left",
                 pad=26 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    va="bottom", ha="left", color=INK_2, fontsize=10.5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_history(path, key="log_history"):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d, d[key]


# ---------------------------------------------------------------- chart 1
def chart_overfitting():
    """The most important chart: eval loss bottoms at epoch 3, then diverges."""
    _, hist = load_history(ROOT / "qwen3-fa-only-run2" / "train_log.json")
    tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(*zip(*tr), color=BLUE, lw=2, label="Training loss")
    ax.plot(*zip(*ev), color=ORANGE, lw=2, marker="o", ms=7, label="Validation loss")

    best = min(ev, key=lambda p: p[1])
    ax.scatter([best[0]], [best[1]], s=150, facecolor=SURFACE, edgecolor=ORANGE,
               zorder=5, linewidth=2.5)
    # Annotations live in the empty mid-right band; the curves occupy top and bottom.
    ax.annotate(f"validation bottoms out\nepoch {best[0]:.0f}, loss {best[1]:.2f}",
                xy=best, xytext=(6.7, 2.05), color=INK_2, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=1.2,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(6.7, 1.30, "after that: training loss keeps falling\nwhile validation rises, memorising",
            color=INK_2, fontsize=10)

    style(ax, "Persian only run overfits after epoch 3",
          "Qwen3-4B LoRA r=64 · 90 Persian examples · 12 epochs · 2× Tesla T4",
          "Epoch", "Loss")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_xlim(0, 12.4)
    fig.tight_layout()
    fig.savefig(OUT / "01-overfitting.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 01-overfitting.png")


# ---------------------------------------------------------------- chart 2
def chart_rhyme():
    """Magnitude comparison -> horizontal bars, every bar directly labelled."""
    rows = [("Training data: all 100 Persian examples (target)", 89, AQUA),
            ("Run 2: Persian only, 12 epochs, r=64", 65, BLUE),
            ("Run 1: 3 languages, 3 epochs, r=16", 33, ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], height=0.42)
    for y, (label, v, _) in zip(ys, rows):
        ax.text(v + 1.5, y, f"{v}%", va="center", color=INK, fontsize=12, fontweight="bold")
        ax.text(0, y - 0.32, label, va="bottom", color=INK_2, fontsize=10)
    ax.set_yticks([]); ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.45, -0.72)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    style(ax, "How much of the poetic form the model actually learned",
          "Share of couplets that rhyme correctly, Persian held out prompts", "Rhyming couplets (%)")
    fig.tight_layout()
    fig.savefig(OUT / "02-rhyme-rate.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 02-rhyme-rate.png")


# ---------------------------------------------------------------- chart 3
def chart_decoding():
    rows = [("No repetition penalty, t=0.7", 53, BLUE),
            ("Greedy, no penalty", 41, ORANGE),
            ("t=0.3, penalty 1.05", 40, ORANGE),
            ("Penalty 1.1, t=0.7  (original)", 39, ORANGE),
            ("t=0.3, no penalty", 36, ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], height=0.40)
    for y, (label, v, _) in zip(ys, rows):
        ax.text(v + 1, y, f"{v}%", va="center", color=INK, fontsize=11, fontweight="bold")
        ax.text(0, y - 0.30, label, va="bottom", color=INK_2, fontsize=10)
    ax.set_yticks([]); ax.set_xlim(0, 62); ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.45, -0.70)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    style(ax, "Repetition penalty fights Persian rhyme",
          "Masnavi needs recurring sounds, penalising repetition breaks it",
          "Rhyming couplets (%)")
    fig.tight_layout()
    fig.savefig(OUT / "03-decoding-sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 03-decoding-sweep.png")


# ---------------------------------------------------------------- chart 4
def chart_termination():
    rows = [("With repetition penalty 1.15", 25, BLUE), ("Greedy decoding", 0, ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], height=0.34)
    for y, (label, v, _) in zip(ys, rows):
        ax.text(v + 0.5, y, f"{v} of 30", va="center", color=INK, fontsize=12, fontweight="bold")
        ax.text(0, y - 0.26, label, va="bottom", color=INK_2, fontsize=10)
    ax.set_yticks([]); ax.set_xlim(0, 32); ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.45, -0.62)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    style(ax, "SmolLM2: the model could stop, the decoder never let it",
          "Held out prompts that terminated on their own instead of looping to the token cap",
          "Prompts that terminated")
    fig.tight_layout()
    fig.savefig(OUT / "04-termination.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote 04-termination.png")


# ---------------------------------------------------------------- gif 1
def gif_training():
    _, hist = load_history(ROOT / "qwen3-fa-only-run2" / "train_log.json")
    tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(9, 5))
    style(ax, "Training the Persian only model",
          "Qwen3-4B LoRA · 90 examples · 12 epochs · 2× Tesla T4 on Kaggle", "Epoch", "Loss")
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.4)
    l_tr, = ax.plot([], [], color=BLUE, lw=2, label="Training loss")
    l_ev, = ax.plot([], [], color=ORANGE, lw=2, marker="o", ms=7, label="Validation loss")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    note = ax.text(0.98, 0.06, "", transform=ax.transAxes, ha="right",
                   color=INK_2, fontsize=11)

    hold = 14
    frames = len(tr) + hold

    def update(i):
        k = min(i, len(tr))
        l_tr.set_data([p[0] for p in tr[:k]], [p[1] for p in tr[:k]])
        e = [p for p in ev if p[0] <= (tr[k-1][0] if k else 0)]
        l_ev.set_data([p[0] for p in e], [p[1] for p in e])
        if k:
            note.set_text(f"epoch {tr[k-1][0]:4.1f}   training loss {tr[k-1][1]:.2f}")
        if i >= len(tr) + 4:
            note.set_text("validation rose from 2.43 to 3.71: memorising, not generalising")
            note.set_color(ORANGE)
        return l_tr, l_ev, note

    anim = FuncAnimation(fig, update, frames=frames, interval=90, blit=False)
    anim.save(OUT / "05-training.gif", writer=PillowWriter(fps=11))
    plt.close(fig)
    print("wrote 05-training.gif")


# ---------------------------------------------------------------- gif 2
def gif_progress():
    """The project's arc, as measured: three stages of Persian rhyme fidelity."""
    stages = [("Run 1\n3 languages, 3 epochs", 33, ORANGE),
              ("Run 2\nPersian only, 12 epochs", 65, BLUE),
              ("Training data\n(the target)", 89, AQUA)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style(ax, "Closing the gap to the training data",
          "Persian rhyming couplets on held out prompts", None, "Rhyming couplets (%)")
    ax.set_ylim(0, 100); ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([s[0] for s in stages], fontsize=10, color=INK_2)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    bars = ax.bar(range(3), [0, 0, 0], color=[s[2] for s in stages], width=0.5)
    labels = [ax.text(i, 2, "", ha="center", color=INK, fontsize=13, fontweight="bold")
              for i in range(3)]

    per, hold = 22, 16
    frames = per * 3 + hold

    def update(f):
        for i, (_, target, _) in enumerate(stages):
            start = i * per
            prog = 0.0 if f < start else min(1.0, (f - start) / per)
            eased = 1 - (1 - prog) ** 3
            v = target * eased
            bars[i].set_height(v)
            labels[i].set_position((i, v + 2.5))
            labels[i].set_text(f"{v:.0f}%" if prog > 0 else "")
        return list(bars) + labels

    anim = FuncAnimation(fig, update, frames=frames, interval=55, blit=False)
    anim.save(OUT / "06-progress.gif", writer=PillowWriter(fps=18))
    plt.close(fig)
    print("wrote 06-progress.gif")


if __name__ == "__main__":
    chart_overfitting()
    chart_rhyme()
    chart_decoding()
    chart_termination()
    gif_training()
    gif_progress()
    print(f"\nall assets in {OUT}")
