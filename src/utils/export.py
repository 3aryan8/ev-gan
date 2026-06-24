# ==============================================================================
# Results Export Utilities
# ==============================================================================
# Collects SOH prediction metrics (RMSE, MAE) and MMD metrics and writes them
# to two output files:
#
#   artifacts/metrics/results.csv   — machine-readable, full float precision
#   artifacts/tables/results.tex    — LaTeX booktabs table, formatted for papers
#
# The exporter is intentionally decoupled from training and evaluation code:
# it receives plain dicts and is responsible only for serialisation.
# ==============================================================================

import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------------------

def export_csv(
    results: Dict[str, Any],
    save_path: str = "artifacts/metrics/results.csv",
) -> None:
    """
    Exports all metrics to a CSV file with full floating-point precision.

    Each row corresponds to one model configuration.  All values are written
    as-is (unrounded) — rounding is the responsibility of the LaTeX exporter.

    Args:
        results:   Nested dict structured as::

                       {
                           "baseline":   {"rmse": float, "mae": float},
                           "pretrained": {"rmse": float, "mae": float},
                           "mmd":        {"mmd2": float, "gamma": float},
                       }

                   Additional top-level keys are tolerated and exported as
                   extra columns.
        save_path: Destination file path. Directories are created if absent.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # Flatten nested dict into rows: {model_name -> {metric -> value}}
    rows = []
    for model_name, metrics in results.items():
        if isinstance(metrics, dict):
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

    if not rows:
        log.warning("[Export] No results to export.")
        return

    # Determine union of all column names (preserving insertion order)
    fieldnames = ["model"]
    for row in rows:
        for k in row:
            if k != "model" and k not in fieldnames:
                fieldnames.append(k)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"[Export] CSV saved → {save_path}  ({len(rows)} rows)")


# ------------------------------------------------------------------------------
# LaTeX export
# ------------------------------------------------------------------------------

_COLUMN_LABELS: Dict[str, str] = {
    "model":    "Model",
    "rmse":     r"RMSE $\downarrow$",
    "mae":      r"MAE $\downarrow$",
    "mmd2":     r"MMD$^2$ $\downarrow$",
    "gamma":    r"$\gamma$ (heuristic)",
}

_DISPLAY_NAMES: Dict[str, str] = {
    "baseline":   r"Baseline (Real-only)",
    "pretrained": r"PhysGAN (Syn.$\to$Real)",
    "mmd":        "Latent MMD",
}

_FLOAT_FMT = "{:.4f}"


def _fmt(value: Any) -> str:
    """Format a cell value: floats to 4 d.p., everything else as str."""
    if isinstance(value, float):
        return _FLOAT_FMT.format(value)
    return str(value)


def export_latex(
    results: Dict[str, Any],
    save_path: str = "artifacts/tables/results.tex",
    caption: str = "Downstream SOH prediction and latent-space MMD results.",
    label: str = "tab:results",
) -> None:
    """
    Exports all metrics as a publication-ready LaTeX ``booktabs`` table.

    The table is self-contained and can be ``\\input{}`` directly into a paper.
    Floating-point values are formatted to 4 decimal places.

    Args:
        results:   Same nested dict as accepted by ``export_csv``.
        save_path: Destination ``.tex`` file path. Directories are created.
        caption:   LaTeX table caption string.
        label:     LaTeX ``\\label`` reference key.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # Determine columns from first row that has metric keys
    metric_cols = []
    for metrics in results.values():
        if isinstance(metrics, dict):
            for k in metrics:
                if k not in metric_cols:
                    metric_cols.append(k)

    all_cols = ["model"] + metric_cols
    n_cols = len(all_cols)
    col_spec = "l" + "r" * len(metric_cols)

    # Build header labels
    header_cells = [_COLUMN_LABELS.get(c, c.replace("_", r"\_")) for c in all_cols]
    header_line = " & ".join(header_cells) + r" \\"

    # Build data rows
    data_lines = []
    for model_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        display_name = _DISPLAY_NAMES.get(model_name, model_name)
        cells = [display_name] + [_fmt(metrics.get(c, "—")) for c in metric_cols]
        data_lines.append(" & ".join(cells) + r" \\")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"% Auto-generated by src/utils/export.py — {timestamp}",
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {header_line}",
        r"    \midrule",
    ]
    for dl in data_lines:
        lines.append(f"    {dl}")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    log.info(f"[Export] LaTeX table saved → {save_path}")


# ------------------------------------------------------------------------------
# Convenience: export both formats in one call
# ------------------------------------------------------------------------------

def export_results(
    soh_metrics: Dict[str, Dict[str, float]],
    mmd_metrics: Dict[str, float],
    metrics_dir: str = "artifacts/metrics",
    tables_dir: str = "artifacts/tables",
    caption: str = "Downstream SOH prediction and latent-space MMD results.",
    label: str = "tab:results",
) -> None:
    """
    Combines SOH and MMD metrics into a single results dict and exports both
    CSV and LaTeX in one call.

    Args:
        soh_metrics:  Output of ``run_downstream_soh`` —
                      ``{"baseline": {"rmse": …, "mae": …}, "pretrained": {…}}``.
        mmd_metrics:  Output of ``evaluate_mmd`` —
                      ``{"mmd2": float, "gamma": float}``.
        metrics_dir:  Directory for the CSV file.
        tables_dir:   Directory for the LaTeX file.
        caption:      LaTeX table caption.
        label:        LaTeX label key.
    """
    results = {**soh_metrics, "mmd": mmd_metrics}

    export_csv(results,   save_path=os.path.join(metrics_dir, "results.csv"))
    export_latex(results, save_path=os.path.join(tables_dir,  "results.tex"),
                 caption=caption, label=label)
