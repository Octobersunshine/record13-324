import io
import json
import uuid
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def _parse_times(time_data):
    parsed = []
    for t in time_data:
        if isinstance(t, (int, float)):
            parsed.append(datetime.fromtimestamp(t))
        else:
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    parsed.append(datetime.strptime(str(t), fmt))
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Cannot parse time: {t}")
    return parsed


def generate_area_chart(
    times,
    series_dict,
    stacked=False,
    title="面积图",
    xlabel="时间",
    ylabel="数值",
    alpha=0.6,
    figsize=(12, 6),
    dpi=150,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    parsed_times = _parse_times(times)
    x = mdates.date2num(parsed_times)

    series_names = list(series_dict.keys())
    series_values = [np.array(series_dict[name], dtype=float) for name in series_names]
    n = len(parsed_times)

    for sv in series_values:
        if len(sv) != n:
            raise ValueError(
                f"Series length {len(sv)} does not match time length {n}"
            )

    if stacked:
        has_negative = any(np.any(sv < 0) for sv in series_values)
        if has_negative:
            pos_vals = [np.maximum(sv, 0) for sv in series_values]
            neg_vals = [np.minimum(sv, 0) for sv in series_values]

            ax.stackplot(
                x,
                *pos_vals,
                labels=series_names,
                colors=PALETTE[: len(series_names)],
                alpha=alpha,
                baseline="zero",
            )
            ax.stackplot(
                x,
                *neg_vals,
                labels=[""] * len(series_names),
                colors=PALETTE[: len(series_names)],
                alpha=alpha * 0.85,
                baseline="zero",
            )
        else:
            ax.stackplot(
                x,
                *series_values,
                labels=series_names,
                colors=PALETTE[: len(series_names)],
                alpha=alpha,
                baseline="zero",
            )
    else:
        for i, (name, values) in enumerate(zip(series_names, series_values)):
            color = PALETTE[i % len(PALETTE)]
            ax.fill_between(
                x,
                0,
                values,
                where=(values >= 0),
                alpha=alpha,
                label=name,
                color=color,
                interpolate=True,
            )
            ax.fill_between(
                x,
                0,
                values,
                where=(values < 0),
                alpha=alpha * 0.85,
                color=color,
                interpolate=True,
            )

    ax.axhline(y=0, color="#333", linewidth=0.8, linestyle="-")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_from_csv_data(csv_text, time_col, value_cols, stacked=False, **kwargs):
    import csv

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    times = [row[time_col] for row in rows]
    series_dict = {col: [float(row[col]) for row in rows] for col in value_cols}

    return generate_area_chart(times, series_dict, stacked=stacked, **kwargs)


def generate_from_json_data(json_str, stacked=False, **kwargs):
    data = json.loads(json_str)

    times = data["times"]
    series_dict = data["series"]

    return generate_area_chart(times, series_dict, stacked=stacked, **kwargs)
