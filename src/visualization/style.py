"""Shared chart style and readable labels for every project notebook.

The visual rules live in ``docs/visualization-rubric.md``; this module is the
single place where notebooks pick up those colours, fonts and feature names.
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter, ConciseDateFormatter, AutoDateFormatter
from matplotlib.ticker import FuncFormatter, PercentFormatter, ScalarFormatter
import numpy as np

PAPER, INK, MUTED, GRID = "#FAFBFC", "#17212B", "#667085", "#DDE3EA"
BLUE, TEAL, CORAL, GOLD, VIOLET = "#2F6BFF", "#16A085", "#E76F51", "#E9A23B", "#7357C8"
CYAN = "#0099CC"
SPLIT_COLORS = {"train": "#6F4EAF", "validation": "#E17C05", "test": "#00A6D6"}
CATEGORY_PALETTE = [
    "#2F6BFF", "#00AA00", "#FF0011", "#FF00AA", "#775522",
    "#0099CC", "#884477", "#008866", "#DD7700", "#AA11DD",
    "#AA0033", "#999900", "#0055AA", "#CC7777", "#336600",
    "#CC66DD", "#993300", "#9988CC", "#FF2277", "#889955",
]
MODEL_COLORS = {
    "Ridge": BLUE,
    "Logistic regression": BLUE,
    "Decision tree": TEAL,
    "Random Forest": VIOLET,
    "Boosting · squared error": GOLD,
    "Boosting · absolute error": "#00AA00",
    "Boosting · Poisson": CYAN,
}

FEATURE_LABELS = {
    "sales": "Current-week sales",
    **{f"lag_{lag}_sales": f"Sales {lag} week{'s' if lag > 1 else ''} ago"
       for lag in range(1, 13)},
    "rolling_4w_sales_mean": "4-week mean sales",
    "rolling_4w_sales_std": "4-week sales variability",
    "rolling_4w_active_weeks": "Active weeks in 4-week history",
    "rolling_12w_sales_mean": "12-week mean sales",
    "rolling_12w_sales_std": "12-week sales variability",
    "rolling_12w_active_weeks": "Active weeks in 12-week history",
    "sales_trend_4w": "4-week sales trend",
    "mean_price": "Current-week mean price",
    "rolling_4w_mean_price": "4-week mean price",
    "price_change_from_4w_mean": "Price change versus 4-week mean",
    "online_share": "Current-week online share",
    "rolling_4w_online_share": "4-week online share",
    "iso_week": "ISO week",
    "week_sin": "Seasonal week (sine)",
    "week_cos": "Seasonal week (cosine)",
    "product_type_name": "Product type",
    "product_group_name": "Product group",
    "index_group_name": "Index group",
    "section_name": "Section",
    "garment_group_name": "Garment group",
    "colour_group_name": "Colour group",
    "target_next_week_sales": "Next-week sales",
    "target_high_demand": "High next-week demand",
}


def feature_label(feature):
    return FEATURE_LABELS[feature]


TITLE_SIZE = 15


def _localize_numeric_text(text):
    """Swap Python's numeric separators without touching non-numeric content."""
    return text.replace(",", "\N{UNIT SEPARATOR}").replace(".", ",").replace(
        "\N{UNIT SEPARATOR}", "."
    )


def format_number(value, decimals=None, signed=False):
    """Format a number with ``.`` for thousands and ``,`` for decimals."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return str(value)
    if decimals is None:
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
            value = int(value)  # 3.0 reads as "3", not "3,0".
            rendered = format(value, ",")
        else:
            rendered = format(value, ",.3f").rstrip("0").rstrip(".")
    else:
        specification = f"{('+' if signed else '')},.{decimals}f"
        rendered = format(value, specification)
    return _localize_numeric_text(rendered)


def format_percent(value, decimals=1):
    """Format a ratio as a percentage using Spanish numeric separators."""
    return f"{format_number(100 * value, decimals)} %"


class _SpanishNumber:
    def __init__(self, value):
        self.value = value

    def __format__(self, specification):
        if specification.endswith("%"):
            decimals = specification.partition(".")[2].removesuffix("%")
            return format_percent(self.value, int(decimals) if decimals else 0)
        return _localize_numeric_text(format(self.value, specification))


def es(value):
    """Allow localized values inside f-strings while preserving their precision."""
    return _SpanishNumber(value)


def display_es(value):
    """Display pandas objects with localized numbers, including numeric indexes."""
    from IPython.display import display
    from pandas.api.types import is_numeric_dtype

    if hasattr(value, "data") and value.__class__.__name__ == "Styler":
        numeric_columns = {
            position for position, dtype in enumerate(value.data.dtypes)
            if is_numeric_dtype(dtype)
        }
        for key, formatter in value._display_funcs.items():
            if key[1] in numeric_columns:
                value._display_funcs[key] = (
                    lambda item, formatter=formatter:
                    _localize_numeric_text(formatter(item))
                )
    elif hasattr(value, "style"):
        value = value.style.format(format_number).format_index(format_number, axis=0)
        if getattr(value.data.index, "nlevels", 1) > 1:
            value = value.format_index(format_number, axis=0, level=None)
    display(value)


def localize_axis_numbers(ax):
    """Apply Spanish separators to numeric and percentage ticks on one axis."""
    date_formatters = (DateFormatter, ConciseDateFormatter, AutoDateFormatter)
    for axis in (ax.xaxis, ax.yaxis):
        formatter = axis.get_major_formatter()
        if isinstance(formatter, PercentFormatter):
            maximum = formatter.xmax
            decimals = formatter.decimals
            axis.set_major_formatter(FuncFormatter(
                lambda value, _position, maximum=maximum, decimals=decimals:
                format_percent(value / maximum, 0 if decimals is None else decimals)
            ))
        elif isinstance(formatter, ScalarFormatter) and not isinstance(formatter, date_formatters):
            axis.set_major_formatter(FuncFormatter(lambda value, _position: format_number(value)))


def add_title(fig, title):
    """Main figure title; constrained layout keeps the same gap above every chart."""
    if fig.get_layout_engine() is None:
        fig.set_layout_engine("constrained")
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold")


def add_source(fig, text):
    """Place the single source note at the bottom-left corner of the figure."""
    for ax in fig.axes:
        localize_axis_numbers(ax)
    engine = fig.get_layout_engine()
    if engine is not None and "rect" in engine.get():
        # Constrained layout would otherwise put the axis labels on the same line.
        left, bottom, width, height = engine.get()["rect"]
        reserved = 0.28 / fig.get_figheight()
        engine.set(rect=(left, bottom + reserved, width, height - reserved))
    fig.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=8, color=MUTED)


def apply_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ["Arial", "Liberation Sans", "DejaVu Sans"]
                 if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font, "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.labelsize": 10, "axes.labelweight": "bold",
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 8,
        "figure.constrained_layout.h_pad": 0.1,
        "figure.titlesize": 15, "figure.titleweight": "bold",
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 9, "legend.facecolor": "white",
        "legend.edgecolor": GRID, "legend.fancybox": True, "legend.framealpha": 1,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    })
    try:
        # Each inline chart also carries an SVG copy, so VS Code can save it as .svg.
        from IPython import get_ipython
        from matplotlib_inline.config import InlineBackend
        if get_ipython() is not None:
            # Stored in the inline-backend config, so the kernel keeps it when it
            # re-applies its matplotlib setup on the first displayed chart.
            InlineBackend.instance().figure_formats = {"png", "svg"}
    except ImportError:
        pass
