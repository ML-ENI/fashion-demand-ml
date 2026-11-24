"""Boxed labels placed next to the lines of a line chart, without covering any line or another label."""

import numpy as np


def place_line_labels(ax, x, curves, texts, colours, fontsize=8):
    """Add one boxed label per line, in the free spot closest to a part of its own line that no other line shares.

    Boxes never cover a line or another box. Each leader goes to the nearest distinctive point of its line and is
    penalised when it crosses another line (lines running alongside it, closer than 4 units, are not counted).
    Call it after the axis limits are final.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    dense_x = np.linspace(x[0], x[-1], 400)
    dense = np.array([np.interp(dense_x, x, curve) for curve in curves])
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    sx, sy = 2 / (x1 - x0), 1 / (y1 - y0)                 # distances in axis fractions; x counts double (wide axes)
    boxes = []
    for i in range(len(curves)):
        probe = ax.text(0, 0, texts[i], fontsize=fontsize, bbox={"boxstyle": "round,pad=0.3"})
        extent = probe.get_window_extent(renderer)
        probe.remove()
        (a, b), (c, d) = to_data.transform([[0, 0], [extent.width, extent.height]])
        w, h = c - a, d - b
        others = np.delete(dense, i, axis=0)
        distinctive = np.abs(others - dense[i]).min(axis=0) >= 4
        anchors = np.flatnonzero(distinctive) if distinctive.any() else np.arange(len(dense_x))
        best = None
        for cx in np.linspace(x0 + w / 2 + 0.05, x1 - w / 2 - 0.05, 60):
            for cy in np.linspace(y0 + h / 2 + 0.5, y1 - h / 2 - 0.5, 60):
                covered = (np.abs(dense_x - cx) <= w / 2 + 0.08)[None, :] & (np.abs(dense - cy) <= h / 2 + 1.0)
                if covered.any() or any(abs(cx - bx) < (w + bw) / 2 + 0.1 and abs(cy - by) < (h + bh) / 2 + 1
                                        for bx, by, bw, bh in boxes):
                    continue
                distance = np.hypot((dense_x[anchors] - cx) * sx, (dense[i][anchors] - cy) * sy)
                k = anchors[int(np.argmin(distance))]
                t = np.linspace(0, 1, 30)
                lx, ly = cx + (dense_x[k] - cx) * t, cy + (dense[i][k] - cy) * t
                crossing = sum(np.any(np.abs(np.interp(lx, dense_x, other) - ly) < 1.0) for other in others
                               if abs(other[k] - dense[i][k]) >= 4)
                score = distance.min() + 0.3 * crossing
                if best is None or score < best[0]:
                    best = (score, cx, cy, k)
        if best is None:
            raise ValueError(f"No free space for the label of line {i}; enlarge the axes.")
        _, cx, cy, k = best
        boxes.append((cx, cy, w, h))
        ax.annotate(texts[i], (dense_x[k], dense[i][k]), xytext=(cx, cy), ha="center", va="center", fontsize=fontsize,
                    color=colours[i], zorder=6,
                    bbox={"facecolor": "white", "edgecolor": colours[i], "boxstyle": "round,pad=0.3", "linewidth": 0.8},
                    arrowprops={"arrowstyle": "-", "color": colours[i], "linewidth": 0.9, "shrinkA": 0, "shrinkB": 1})
