import os
import json
import ast
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "analytics")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BTP_TO_OUR_MAP = {
    "WRONG PARKING": "ILLEGAL_PARKING",
    "NO PARKING": "ILLEGAL_PARKING",
    "PARKING": "ILLEGAL_PARKING",
    "DEFECTIVE NUMBER PLATE": "PLATE_VIOLATION",
    "NO NUMBER PLATE": "PLATE_VIOLATION",
    "NUMBER PLATE": "PLATE_VIOLATION",
    "WITHOUT HELMET": "HELMET_VIOLATION",
    "NO HELMET": "HELMET_VIOLATION",
    "HELMET": "HELMET_VIOLATION",
    "NO SEAT BELT": "NO_SEATBELT",
    "SEAT BELT": "NO_SEATBELT",
    "TRIPLE RIDING": "TRIPLE_RIDING",
    "TRIPLE LOAD": "TRIPLE_RIDING",
    "STOP LINE": "STOPLINE_VIOLATION",
    "STOPLINE": "STOPLINE_VIOLATION",
    "STOP LINE VIOLATION": "STOPLINE_VIOLATION",
    "RED LIGHT": "RED_LIGHT_VIOLATION",
    "RED LIGHT JUMP": "RED_LIGHT_VIOLATION",
    "SIGNAL JUMP": "RED_LIGHT_VIOLATION",
    "SIGNAL": "RED_LIGHT_VIOLATION",
    "SPEEDING": "SPEEDING",
    "OVER SPEED": "SPEEDING",
    "WRONG WAY": "WRONG_WAY",
    "ONE WAY": "WRONG_WAY",
    "LANE VIOLATION": "LANE_VIOLATION",
    "LANE CUTTING": "LANE_VIOLATION",
    "DANGEROUS DRIVING": "RECKLESS_DRIVING",
    "RECKLESS DRIVING": "RECKLESS_DRIVING",
    "DRINKING": "DRUNK_DRIVING",
    "DRUNK DRIVING": "DRUNK_DRIVING",
    "DRUNK AND DRIVE": "DRUNK_DRIVING",
    "NO INSURANCE": "DOCUMENT_VIOLATION",
    "INSURANCE": "DOCUMENT_VIOLATION",
    "NO LICENSE": "DOCUMENT_VIOLATION",
    "LICENSE": "DOCUMENT_VIOLATION",
    "NO DOCUMENT": "DOCUMENT_VIOLATION",
    "POLLUTION": "POLLUTION_VIOLATION",
    "PUC": "POLLUTION_VIOLATION",
    "OBSTRUCTION": "OBSTRUCTION",
    "U TURN": "WRONG_WAY",
    "ILLEGAL U TURN": "WRONG_WAY",
    "OVERLOADING": "OVERLOADING",
    "OVER LOAD": "OVERLOADING",
    "MOBILE": "MOBILE_USAGE",
    "MOBILE PHONE": "MOBILE_USAGE",
    "HORN": "HORN_PROHIBITION",
    "HORN PROHIBITION": "HORN_PROHIBITION",
    "TINTED GLASS": "TINTED_GLASS",
    "INDECENT": "INDECENCY",
    "OBSCENE": "INDECENCY",
}

OUR_DETECTABLE = {
    "ILLEGAL_PARKING", "PLATE_VIOLATION", "HELMET_VIOLATION",
    "NO_SEATBELT", "TRIPLE_RIDING", "STOPLINE_VIOLATION",
    "RED_LIGHT_VIOLATION", "SPEEDING", "WRONG_WAY", "LANE_VIOLATION",
}

COLORS = plt.cm.Set3(np.linspace(0, 1, 15))


def parse_violation_types(series):
    parsed = []
    failed = 0
    for val in series:
        if pd.isna(val) or val == "" or val == "[]":
            continue
        try:
            items = ast.literal_eval(val)
            if isinstance(items, list):
                parsed.extend(items)
            else:
                parsed.append(str(items))
        except (ValueError, SyntaxError, MemoryError):
            failed += 1
    return parsed, failed


def load_and_parse(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} rows, columns: {list(df.columns)}")
    date_col = None
    for candidate in ["date", "timestamp", "date_time", "violation_date", "issued_date"]:
        if candidate in df.columns:
            date_col = candidate
            break
    parsed_types, failed = parse_violation_types(df["violation_type"])
    print(f"Parsed {len(parsed_types)} individual violations "
          f"({failed} rows failed to parse)")
    counts = Counter(parsed_types)
    top_30 = counts.most_common(30)
    print(f"\nUnique violation types found: {len(counts)}")
    print(f"{'#':>3}  {'Type':<35}  {'Count':>8}")
    print("-" * 50)
    for i, (vt, cnt) in enumerate(top_30, 1):
        print(f"{i:>3}  {vt:<35}  {cnt:>8,}")
    return df, parsed_types, counts, date_col


def build_mapped_counts(counts: Counter):
    mapped = Counter()
    unmapped = Counter()
    for vt, cnt in counts.items():
        normalized = vt.strip().upper()
        if normalized in BTP_TO_OUR_MAP:
            mapped[BTP_TO_OUR_MAP[normalized]] += cnt
        else:
            unmapped[vt] = cnt
    return mapped, unmapped


def calc_coverage(mapped: Counter, unmapped: Counter):
    total = sum(mapped.values()) + sum(unmapped.values())
    our_total = sum(mapped[k] for k in mapped if k in OUR_DETECTABLE)
    covered_pct = round(our_total / total * 100, 2) if total else 0
    uncovered_pct = round(100 - covered_pct, 2)
    print(f"\n{'='*45}")
    print(f"  Coverage Analysis")
    print(f"{'='*45}")
    print(f"  Total violations parsed: {total:,}")
    print(f"  Our system can detect:   {our_total:,} ({covered_pct}%)")
    print(f"  Outside our scope:       {total - our_total:,} ({uncovered_pct}%)")
    print(f"{'='*45}")
    return our_total, total, covered_pct, uncovered_pct


def plot_top_15(counts: Counter, output_dir: str):
    top = counts.most_common(15)
    labels, values = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), values, color=COLORS[:len(labels)],
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title("Top 15 BTP Violation Types", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7)
    ax.margins(x=0.1)
    plt.tight_layout()
    path = os.path.join(output_dir, "btp_top15_violations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_coverage_pie(mapped: Counter, output_dir: str):
    our_sum = sum(mapped[k] for k in mapped if k in OUR_DETECTABLE)
    other_sum = sum(mapped[k] for k in mapped if k not in OUR_DETECTABLE)
    fig, ax = plt.subplots(figsize=(7, 7))
    sizes = [our_sum, other_sum]
    labels_display = [
        f"Detectable by Our System\n{our_sum:,} ({our_sum/sum(sizes)*100:.1f}%)",
        f"Not Detectable\n{other_sum:,} ({other_sum/sum(sizes)*100:.1f}%)",
    ]
    colors_pie = ["#22c55e", "#ef4444"]
    explode = (0.03, 0.03)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="", startangle=90,
        explode=explode, colors=colors_pie, pctdistance=0.8,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    legend_handles = [
        mpatches.Patch(color="#22c55e", label=labels_display[0]),
        mpatches.Patch(color="#ef4444", label=labels_display[1]),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)
    ax.set_title("Violation Coverage: Our System vs Uncovered",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "btp_coverage_pie.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_monthly_trend(df: pd.DataFrame, date_col: str, output_dir: str):
    if date_col is None:
        print("  Skipping monthly trend — no date column found.")
        return
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_clean = df_copy.dropna(subset=[date_col])
        if df_clean.empty:
            print("  Skipping monthly trend — no valid dates.")
            return
        df_clean["month"] = df_clean[date_col].dt.to_period("M").astype(str)
        monthly = df_clean.groupby("month").size()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(monthly)), monthly.values, marker="o",
                color="#facc15", linewidth=2, markersize=5)
        ax.fill_between(range(len(monthly)), monthly.values,
                         alpha=0.15, color="#facc15")
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(monthly.index, rotation=45, fontsize=8)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Violations", fontsize=10)
        ax.set_title("Monthly Violation Trend", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "btp_monthly_trend.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"  Skipping monthly trend plot: {e}")


def plot_heatmap(df: pd.DataFrame, counts: Counter, output_dir: str):
    top_types = [t for t, _ in counts.most_common(10)]
    if len(df) < 100:
        print("  Skipping heatmap — too few records.")
        return
    df_subset = df.head(5000).copy()
    type_matrix = defaultdict(lambda: defaultdict(int))
    failed_rows = 0
    for _, row in df_subset.iterrows():
        val = row.get("violation_type")
        if pd.isna(val) or val == "" or val == "[]":
            continue
        try:
            items = ast.literal_eval(val)
            if not isinstance(items, list):
                continue
            items_set = set(items)
            for i, t1 in enumerate(items):
                if t1 not in top_types:
                    continue
                for t2 in list(items_set)[:]:
                    if t2 not in top_types or t1 >= t2:
                        continue
                    type_matrix[t1][t2] += 1
        except (ValueError, SyntaxError):
            failed_rows += 1
    if not type_matrix:
        print("  Skipping heatmap — no co-occurrence data.")
        return
    type_labels = [t for t in top_types if
                   t in type_matrix or any(t in v for v in type_matrix.values())]
    n = len(type_labels)
    if n < 2:
        print("  Skipping heatmap — fewer than 2 co-occurring types.")
        return
    heatmap_data = np.zeros((n, n))
    for i, t1 in enumerate(type_labels):
        for j, t2 in enumerate(type_labels):
            heatmap_data[i, j] = type_matrix.get(t1, {}).get(t2, 0)
    if heatmap_data.sum() == 0:
        print("  Skipping heatmap — no co-occurrences detected.")
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(type_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(type_labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            val = heatmap_data[i, j]
            if val > 0:
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=6, color="black" if val < heatmap_data.max()/2 else "white")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Co-occurrence count")
    ax.set_title("Top-10 Violation Type Co-occurrence Heatmap",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "btp_cooccurrence_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def export_summary(mapped: Counter, unmapped: Counter,
                   total_records: int, date_range: str, output_dir: str):
    sorted_mapped = mapped.most_common()
    top_5 = [
        {"violation_type": k, "count": v, "mapped_to": BTP_TO_OUR_MAP.get(k, k)}
        for k, v in sorted_mapped[:5]
    ]
    our_total = sum(mapped[k] for k in mapped if k in OUR_DETECTABLE)
    coverage_pct = round(our_total / total_records * 100, 2) if total_records else 0
    summary = {
        "total_records": total_records,
        "date_range": date_range,
        "top_5_violations": top_5,
        "our_coverage_percent": coverage_pct,
        "uncovered_percent": round(100 - coverage_pct, 2),
        "total_mapped": sum(mapped.values()),
        "total_unmapped": sum(unmapped.values()),
        "detectable_count": our_total,
    }
    path = os.path.join(output_dir, "btp_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported: {path}")
    return summary


def print_summary_table(mapped: Counter, unmapped: Counter):
    print(f"\n{'='*60}")
    print(f"  Violation Type Mapping Summary")
    print(f"{'='*60}")
    print(f"  {'Our Class':<25} {'Count':>8}")
    print(f"  {'-'*35}")
    for k, v in sorted(mapped.items(), key=lambda x: -x[1]):
        symbol = "✓" if k in OUR_DETECTABLE else "✗"
        print(f"  {symbol} {k:<25} {v:>8,}")
    if unmapped:
        print(f"\n  Unmapped types ({sum(unmapped.values()):,} total):")
        for k, v in unmapped.most_common(10):
            print(f"    • {k:<40} {v:>8,}")


def main():
    csv_path = os.path.join(
        DATA_DIR, "jan_to_may_police_violation_anonymized.csv"
    )
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Creating sample data for demonstration...")
        sample_types = [
            '["WRONG PARKING"]',
            '["NO PARKING"]',
            '["WITHOUT HELMET"]',
            '["SIGNAL JUMP"]',
            '["NO SEAT BELT"]',
            '["TRIPLE RIDING"]',
            '["WRONG WAY"]',
            '["SPEEDING"]',
            '["NO PARKING","WRONG PARKING"]',
            '["DEFECTIVE NUMBER PLATE"]',
            '["DRUNK DRIVING"]',
            '["NO INSURANCE"]',
            '["OVER SPEED"]',
            '["STOP LINE VIOLATION"]',
            '["LANE CUTTING"]',
            '["OBSTRUCTION"]',
            '["POLLUTION"]',
            '["MOBILE PHONE"]',
            '["HORN PROHIBITION"]',
            '["TINTED GLASS"]',
            '["WITHOUT HELMET","TRIPLE RIDING"]',
            '["NO SEAT BELT","MOBILE PHONE"]',
            '["ONE WAY"]',
            '["DRUNK AND DRIVE"]',
            '["U TURN"]',
            '["NO PARKING","OBSTRUCTION"]',
            '["OVER LOAD"]',
            '["INDECENT"]',
            '["NO LICENSE"]',
            '["RED LIGHT JUMP"]',
        ]
        np.random.seed(42)
        n_rows = 5000
        dates = pd.date_range(start="2025-01-01", end="2025-05-31", periods=n_rows)
        df = pd.DataFrame({
            "violation_type": np.random.choice(sample_types, n_rows),
            "date": dates,
        })
        print(f"Generated {n_rows} sample rows with {len(sample_types)} violation types\n")
        parsed_types, failed = parse_violation_types(df["violation_type"])
        counts = Counter(parsed_types)
        date_col = "date"
        print(f"Parsed {len(parsed_types)} individual violations ({failed} failed)")
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {len(df)} rows, columns: {list(df.columns)}")
        df, parsed_types, counts, date_col = load_and_parse(csv_path)
    mapped, unmapped = build_mapped_counts(counts)
    total_records = sum(counts.values())
    date_range_str = ""
    if date_col and date_col in df.columns:
        try:
            dt_col = pd.to_datetime(df[date_col], errors="coerce")
            dt_valid = dt_col.dropna()
            if not dt_valid.empty:
                date_range_str = f"{dt_valid.min().strftime('%Y-%m-%d')} to {dt_valid.max().strftime('%Y-%m-%d')}"
        except Exception:
            pass
    print(f"\n{'='*60}")
    print(f"  Date range: {date_range_str or 'N/A'}")
    print(f"  Total individual violation tags: {total_records:,}")
    print(f"{'='*60}")
    plot_top_15(counts, OUTPUT_DIR)
    plot_coverage_pie(mapped, OUTPUT_DIR)
    plot_monthly_trend(df, date_col, OUTPUT_DIR)
    plot_heatmap(df, counts, OUTPUT_DIR)
    our_total, total, covered_pct, uncovered_pct = calc_coverage(mapped, unmapped)
    print_summary_table(mapped, unmapped)
    summary = export_summary(
        mapped, unmapped, total_records, date_range_str, OUTPUT_DIR
    )
    print(f"\nDone. All charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
