"""Shared style for the figures: IEEE column widths and serif text."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

COLUMN_WIDTH = 3.5  # in, IEEE two-column layout
TEXT_WIDTH = 7.16

# Series colors and neutral inks.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"

# Background that keeps text labels readable where lines cross them.
LABEL_BOX = {"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 1.0}

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_SECONDARY,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.4,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def style() -> None:
    plt.rcParams.update(STYLE)


def save(fig: plt.Figure, path: Path) -> None:
    """PDF and PNG, without timestamps so that regenerated files are identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
