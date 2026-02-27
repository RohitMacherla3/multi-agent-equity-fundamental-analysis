from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


def _to_float(value: str | float | int | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def _group_by_ticker(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        t = str(row.get("ticker") or "UNKNOWN").strip() or "UNKNOWN"
        out.setdefault(t, []).append(row)
    return out


def _sorted_items_desc(data: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(((k, max(0.0, float(v))) for k, v in data.items()), key=lambda x: x[1], reverse=True)


def _svg_lollipop_chart(title: str, subtitle: str, data: dict[str, float], out_path: Path, color: str, is_percent: bool = False) -> None:
    items = _sorted_items_desc(data)
    w = 980
    h = 430
    left = 180
    right = 48
    top = 82
    row_h = 30
    bottom = 38
    plot_w = w - left - right
    max_v = max([v for _, v in items], default=1.0)
    max_v = max(max_v, 1e-6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="40" fill="#f5f7ff" font-size="26" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left}" y="64" fill="#9aa3be" font-size="14" font-family="Arial">{subtitle}</text>',
    ]
    for i, (ticker, value) in enumerate(items):
        y = top + i * row_h
        x2 = left + (value / max_v) * plot_w
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#1b2742" stroke-width="6" stroke-linecap="round"/>')
        parts.append(f'<line x1="{left}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x2:.1f}" cy="{y}" r="6" fill="{color}" stroke="#dfe9ff" stroke-width="1"/>')
        label = f"{value:.2%}" if is_percent else f"{value:.4g}"
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" fill="#cdd8f3" font-size="13" font-family="Arial">{ticker}</text>')
        parts.append(f'<text x="{left+plot_w+8}" y="{y+5}" fill="#e8efff" font-size="12" font-family="Arial">{label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_dumbbell_latency(title: str, subtitle: str, avg_data: dict[str, float], p95_data: dict[str, float], out_path: Path) -> None:
    tickers = sorted(avg_data.keys(), key=lambda t: avg_data[t], reverse=True)
    w = 980
    h = 430
    left = 180
    right = 70
    top = 86
    row_h = 30
    plot_w = w - left - right
    max_v = max([max(avg_data.get(t, 0.0), p95_data.get(t, 0.0)) for t in tickers], default=1.0)
    max_v = max(max_v, 1e-6)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="40" fill="#f5f7ff" font-size="26" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left}" y="64" fill="#9aa3be" font-size="14" font-family="Arial">{subtitle}</text>',
        f'<text x="{left}" y="78" fill="#7fc8ff" font-size="12" font-family="Arial">● Avg</text>',
        f'<text x="{left+70}" y="78" fill="#8bf2cc" font-size="12" font-family="Arial">● P95</text>',
    ]
    for i, t in enumerate(tickers):
        y = top + i * row_h
        avg_x = left + (avg_data.get(t, 0.0) / max_v) * plot_w
        p95_x = left + (p95_data.get(t, 0.0) / max_v) * plot_w
        x1, x2 = min(avg_x, p95_x), max(avg_x, p95_x)
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" fill="#cdd8f3" font-size="13" font-family="Arial">{t}</text>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#33496a" stroke-width="4"/>')
        parts.append(f'<circle cx="{avg_x:.1f}" cy="{y}" r="5.5" fill="#7fc8ff"/>')
        parts.append(f'<circle cx="{p95_x:.1f}" cy="{y}" r="5.5" fill="#8bf2cc"/>')
        parts.append(f'<text x="{left+plot_w+8}" y="{y+5}" fill="#e8efff" font-size="12" font-family="Arial">{avg_data.get(t, 0.0):.2f}s / {p95_data.get(t, 0.0):.2f}s</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_latency_histogram(title: str, subtitle: str, data: dict[str, float], out_path: Path) -> None:
    # Simple vertical histogram-style bar chart.
    items = _sorted_items_desc(data)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    w = 980
    h = 430
    left = 90
    right = 40
    top = 82
    bottom = 86
    plot_w = w - left - right
    plot_h = h - top - bottom
    n = max(1, len(labels))
    slot = plot_w / n
    bar_w = slot * 0.62
    max_v = max(values) if values else 1.0
    max_v = max(max_v, 1e-6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="40" fill="#f5f7ff" font-size="26" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left}" y="64" fill="#9aa3be" font-size="14" font-family="Arial">{subtitle}</text>',
        f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#32415f" stroke-width="1"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * slot + (slot - bar_w) / 2
        bh = (value / max_v) * (plot_h - 10)
        y = h - bottom - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="6" fill="#7fc8ff"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" fill="#dfe9ff" font-size="12" font-family="Arial">{value:.1f}s</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{h-bottom+22}" text-anchor="middle" fill="#c9d2f0" font-size="12" font-family="Arial">{label}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_ranked_cost(title: str, subtitle: str, cost_data: dict[str, float], out_path: Path) -> None:
    # Lower cost is better: show sorted ascending horizontal bars.
    items = sorted(((k, max(0.0, float(v))) for k, v in cost_data.items()), key=lambda x: x[1])
    w = 980
    h = 430
    left = 190
    right = 60
    top = 92
    row_h = 30
    bar_h = 16
    plot_w = w - left - right
    max_v = max([v for _, v in items], default=1.0)
    max_v = max(max_v, 1e-9)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="40" fill="#f5f7ff" font-size="26" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left}" y="64" fill="#9aa3be" font-size="14" font-family="Arial">{subtitle}</text>',
    ]
    for i, (ticker, v) in enumerate(items):
        y = top + i * row_h
        bw = (v / max_v) * plot_w
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" fill="#cdd8f3" font-size="13" font-family="Arial">{ticker}</text>')
        parts.append(f'<rect x="{left}" y="{y-bar_h/2:.1f}" width="{plot_w:.1f}" height="{bar_h}" rx="6" fill="#1c2945"/>')
        parts.append(f'<rect x="{left}" y="{y-bar_h/2:.1f}" width="{bw:.1f}" height="{bar_h}" rx="6" fill="#7cc8ff"/>')
        parts.append(f'<text x="{left+plot_w+8}" y="{y+4}" fill="#eaf2ff" font-size="12" font-family="Arial">${v:,.6f}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_stacked_tokens(title: str, subtitle: str, input_data: dict[str, float], output_data: dict[str, float], out_path: Path) -> None:
    tickers = sorted(input_data.keys(), key=lambda t: (input_data[t] + output_data.get(t, 0.0)), reverse=True)
    totals = {t: input_data[t] + output_data.get(t, 0.0) for t in tickers}
    max_v = max(totals.values(), default=1.0)
    max_v = max(max_v, 1e-6)
    w = 980
    h = 430
    left = 180
    right = 70
    top = 88
    row_h = 30
    plot_w = w - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="40" fill="#f5f7ff" font-size="26" font-family="Arial" font-weight="700">{title}</text>',
        f'<text x="{left}" y="64" fill="#9aa3be" font-size="14" font-family="Arial">{subtitle}</text>',
        f'<text x="{left}" y="78" fill="#7cc8ff" font-size="12" font-family="Arial">■ Input</text>',
        f'<text x="{left+60}" y="78" fill="#8bf2cc" font-size="12" font-family="Arial">■ Output</text>',
    ]
    for i, t in enumerate(tickers):
        y = top + i * row_h
        in_v = input_data.get(t, 0.0)
        out_v = output_data.get(t, 0.0)
        in_w = (in_v / max_v) * plot_w
        out_w = (out_v / max_v) * plot_w
        parts.append(f'<text x="{left-12}" y="{y+5}" text-anchor="end" fill="#cdd8f3" font-size="13" font-family="Arial">{t}</text>')
        parts.append(f'<rect x="{left}" y="{y-8}" width="{in_w:.1f}" height="16" rx="6" fill="#7cc8ff"/>')
        parts.append(f'<rect x="{left+in_w:.1f}" y="{y-8}" width="{out_w:.1f}" height="16" rx="6" fill="#8bf2cc"/>')
        parts.append(f'<text x="{left+plot_w+8}" y="{y+5}" fill="#eaf2ff" font-size="12" font-family="Arial">{int(in_v):,} + {int(out_v):,}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_signal_board(
    title: str, grouped: dict[str, list[dict[str, str]]], avg_latency: dict[str, float], out_path: Path
) -> None:
    # Heatmap over 3 metrics per ticker with discrete score bands (0-100).
    tickers = sorted(grouped.keys())
    metrics = ["Citation", "Confidence", "Latency"]
    citation = {t: mean(_to_float(r.get("citation_coverage")) for r in grouped[t]) for t in tickers}
    confidence = {t: mean(_to_float(r.get("confidence_score")) for r in grouped[t]) for t in tickers}
    w = 980
    h = 460
    left = 170
    top = 90
    cw = 165
    ch = 28

    def min_max(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 1.0
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-9:
            return lo, lo + 1.0
        return lo, hi

    def norm(v: float, lo: float, hi: float, invert: bool = False) -> float:
        n = (v - lo) / (hi - lo)
        if invert:
            n = 1.0 - n
        return max(0.0, min(1.0, n))

    c_lo, c_hi = min_max([citation.get(t, 0.0) for t in tickers])
    f_lo, f_hi = min_max([confidence.get(t, 0.0) for t in tickers])
    l_lo, l_hi = min_max([avg_latency.get(t, 0.0) for t in tickers])

    def color_for_score(score_0_1: float) -> str:
        # Discrete bins for easier read.
        if score_0_1 < 0.2:
            return "#6b2d2f"  # very low
        if score_0_1 < 0.4:
            return "#7a4b2f"  # low
        if score_0_1 < 0.6:
            return "#6f6431"  # medium
        if score_0_1 < 0.8:
            return "#3f6a44"  # good
        return "#2e6f5a"  # excellent

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="40" fill="#f5f7ff" font-size="26" font-family="Arial" font-weight="700">{title}</text>',
        '<text x="170" y="64" fill="#9aa3be" font-size="14" font-family="Arial">Normalized score heatmap (0-100, higher = better)</text>',
    ]
    legend_x = left + (3 * cw) + 24
    legend_y = 104
    legend_w = 92
    legend_h = 14
    legend_colors = ["#6b2d2f", "#7a4b2f", "#6f6431", "#3f6a44", "#2e6f5a"]
    legend_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    parts.append(f'<text x="{legend_x}" y="{legend_y-10}" fill="#b7c8e8" font-size="12" font-family="Arial" font-weight="700">Legend</text>')
    for i, (c, lbl) in enumerate(zip(legend_colors, legend_labels)):
        y = legend_y + i * 24
        parts.append(f'<rect x="{legend_x}" y="{y}" width="{legend_w}" height="{legend_h}" rx="4" fill="{c}"/>')
        parts.append(f'<text x="{legend_x + legend_w + 10}" y="{y + 11}" fill="#b7c8e8" font-size="11" font-family="Arial">{lbl}</text>')
    for i, m in enumerate(metrics):
        parts.append(f'<text x="{left + i*cw + 8}" y="{top-10}" fill="#cdd8f3" font-size="13" font-family="Arial" font-weight="700">{m}</text>')
    for r, t in enumerate(tickers):
        y = top + r * (ch + 8)
        parts.append(f'<text x="{left-10}" y="{y+19}" text-anchor="end" fill="#cdd8f3" font-size="13" font-family="Arial">{t}</text>')
        raw_vals = [citation.get(t, 0.0), confidence.get(t, 0.0), avg_latency.get(t, 0.0)]
        scores = [
            norm(raw_vals[0], c_lo, c_hi, invert=False),
            norm(raw_vals[1], f_lo, f_hi, invert=False),
            norm(raw_vals[2], l_lo, l_hi, invert=True),
        ]
        labels = [
            f"{raw_vals[0]:.1%}",
            f"{raw_vals[1]:.2f}",
            f"{raw_vals[2]:.1f}s",
        ]
        for c, (m, s, lab) in enumerate(zip(metrics, scores, labels)):
            x = left + c * cw
            fill = color_for_score(s)
            parts.append(f'<rect x="{x}" y="{y}" width="{cw-10}" height="{ch}" rx="6" fill="{fill}" stroke="#1f2c44" stroke-width="1"/>')
            parts.append(f'<text x="{x+10}" y="{y+19}" fill="#eef4ff" font-size="12" font-family="Arial">{lab}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _kpi_sections(metrics: dict[str, float]) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    return [
        (
            "Quality",
            "#153655",
            "#5bb7ff",
            [
                ("Retrieval Hit Rate", f"{metrics.get('retrieval_hit_rate', 0.0):.2%}"),
                ("Citation Coverage", f"{metrics.get('citation_coverage_overall', 0.0):.2%}"),
                ("News Coverage Rate", f"{metrics.get('news_coverage_rate', 0.0):.2%}"),
                ("Success Rate", f"{metrics.get('success_rate', 0.0):.2%}"),
                ("Error Rate", f"{metrics.get('error_rate', 0.0):.2%}"),
            ],
        ),
        (
            "Performance",
            "#3d2e17",
            "#ffcd75",
            [
                ("Request Count", f"{int(metrics.get('request_count', 0))}"),
                ("Avg Cost / Query (USD)", f"${metrics.get('avg_cost_per_query_usd', 0.0):,.6f}"),
                ("Avg Latency (s)", f"{metrics.get('avg_latency_seconds', 0.0):,.2f}"),
                ("P95 Latency (s)", f"{metrics.get('p95_latency_seconds', 0.0):,.2f}"),
                ("Total Evidence Count", f"{metrics.get('total_evidence_count', 0.0):,.0f}"),
            ],
        ),
        (
            "Token Usage",
            "#1f3a2f",
            "#6df0b3",
            [
                ("Avg Input Tokens", f"{metrics.get('avg_input_tokens', 0.0):,.0f}"),
                ("Avg Output Tokens", f"{metrics.get('avg_output_tokens', 0.0):,.0f}"),
            ],
        ),
    ]


def _svg_kpi_dashboard_option_a(metrics: dict[str, float], out_path: Path) -> None:
    # Option A: Dense card grid with section colors.
    w = 980
    sections = _kpi_sections(metrics)
    cards: list[tuple[str, str, str, str]] = []
    for section_name, section_fill, accent, items in sections:
        for label, value in items:
            cards.append((section_name, section_fill, accent, f"{label}|{value}"))

    cols = 4
    cw = 210
    ch = 122
    gapx = 22
    gapy = 22
    start_x = 40
    start_y = 100
    rows = max(1, (len(cards) + cols - 1) // cols)
    h = start_y + rows * ch + (rows - 1) * gapy + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#081126"/>',
        '<text x="40" y="46" fill="#f5f7ff" font-size="33" font-family="Arial" font-weight="700">Evaluation KPI Dashboard - Option A</text>',
        '<text x="40" y="74" fill="#98a4c5" font-size="14" font-family="Arial">Structured grid by metric category</text>',
    ]
    for i, (_, section_fill, accent, payload) in enumerate(cards):
        label, value = payload.split("|", 1)
        r = i // cols
        c = i % cols
        x = start_x + c * (cw + gapx)
        y = start_y + r * (ch + gapy)
        parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="12" fill="{section_fill}" stroke="{accent}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x+14}" y="{y+28}" fill="#c8d4ef" font-size="13" font-family="Arial">{label}</text>')
        parts.append(f'<text x="{x+14}" y="{y+78}" fill="#f8fbff" font-size="34" font-family="Arial" font-weight="700">{value}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_kpi_dashboard_option_b(metrics: dict[str, float], out_path: Path) -> None:
    # Option B: Three vertical sections with stacked cards.
    sections = _kpi_sections(metrics)
    w = 1180
    left = 36
    top = 86
    section_w = 356
    gap = 18
    card_h = 94
    card_gap = 12
    max_cards = max(len(items) for _, _, _, items in sections)
    h = top + 52 + max_cards * card_h + (max_cards - 1) * card_gap + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0d111b"/>',
        '<text x="36" y="44" fill="#f5f7ff" font-size="32" font-family="Arial" font-weight="700">Evaluation KPI Dashboard - Option B</text>',
        '<text x="36" y="68" fill="#9da8c2" font-size="14" font-family="Arial">Section-wise columns: Quality, Performance, Token Usage</text>',
    ]

    for i, (section_name, section_fill, accent, items) in enumerate(sections):
        sx = left + i * (section_w + gap)
        sy = top
        section_h = 52 + len(items) * card_h + (len(items) - 1) * card_gap + 12
        parts.append(f'<rect x="{sx}" y="{sy}" width="{section_w}" height="{section_h}" rx="14" fill="{section_fill}" stroke="{accent}" stroke-width="1.4"/>')
        parts.append(f'<text x="{sx+14}" y="{sy+32}" fill="{accent}" font-size="20" font-family="Arial" font-weight="700">{section_name}</text>')
        for j, (label, value) in enumerate(items):
            cx = sx + 12
            cy = sy + 46 + j * (card_h + card_gap)
            parts.append(f'<rect x="{cx}" y="{cy}" width="{section_w-24}" height="{card_h}" rx="10" fill="#12192b" stroke="#354162" stroke-width="1"/>')
            parts.append(f'<text x="{cx+12}" y="{cy+30}" fill="#9eaccf" font-size="13" font-family="Arial">{label}</text>')
            parts.append(f'<text x="{cx+12}" y="{cy+69}" fill="#f4f7ff" font-size="28" font-family="Arial" font-weight="700">{value}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _svg_kpi_dashboard_option_c(
    metrics: dict[str, float], agent_token_breakdown: dict[str, float], out_path: Path, meta: dict[str, str]
) -> None:
    # Option C: Large top KPIs + quality progress bars + token distribution block.
    w = 1180
    h = 900

    headline = [
        ("Avg Cost / Query", f"${metrics.get('avg_cost_per_query_usd', 0.0):,.6f}"),
        ("Avg Latency", f"{metrics.get('avg_latency_seconds', 0.0):,.2f}s"),
        ("Avg Input Tokens", f"{metrics.get('avg_input_tokens', 0.0):,.0f}"),
        ("Avg Output Tokens", f"{metrics.get('avg_output_tokens', 0.0):,.0f}"),
    ]
    progress = [
        ("Retrieval Hit Rate", float(metrics.get("retrieval_hit_rate", 0.0)), "#64b5ff"),
        ("Citation Coverage", float(metrics.get("citation_coverage_overall", 0.0)), "#85f2c3"),
        ("Success Rate", float(metrics.get("success_rate", 0.0)), "#73d7c7"),
        ("News Coverage", float(metrics.get("news_coverage_rate", 0.0)), "#7de0d2"),
        ("Error Rate", float(metrics.get("error_rate", 0.0)), "#ff8d8d"),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0a0f1f"/>',
        '<rect x="30" y="24" width="1120" height="72" rx="16" fill="#111933" stroke="#394a78" stroke-width="1.3"/>',
        '<text x="54" y="62" fill="#f5f7ff" font-size="30" font-family="Arial" font-weight="700">Evaluation KPI Dashboard</text>',
    ]

    chip_w = 238
    chip_h = 92
    chip_gap = 56
    chip_styles = [
        ("#13343c", "#4fd1c5", "#9ee9df"),  # teal
        ("#162f46", "#63b3ed", "#b8dcff"),  # blue
        ("#1b3a2c", "#68d391", "#b9f3cc"),  # green
        ("#2a2f4d", "#8da2fb", "#cfd7ff"),  # indigo
    ]
    for i, (label, value) in enumerate(headline):
        x = 30 + i * (chip_w + chip_gap)
        y = 124
        fill, stroke, label_color = chip_styles[i % len(chip_styles)]
        parts.append(f'<rect x="{x}" y="{y}" width="{chip_w}" height="{chip_h}" rx="11" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        parts.append(f'<text x="{x+14}" y="{y+30}" fill="{label_color}" font-size="16" font-family="Arial">{label}</text>')
        parts.append(f'<text x="{x+14}" y="{y+76}" fill="#f5f8ff" font-size="39" font-family="Arial" font-weight="700">{value}</text>')

    # Left: quality progress bars
    parts.append('<rect x="30" y="254" width="548" height="556" rx="14" fill="#152536" stroke="#5bb7ff" stroke-width="1.2"/>')
    parts.append('<text x="48" y="289" fill="#5bb7ff" font-size="22" font-family="Arial" font-weight="700">Quality Metrics</text>')
    bar_x = 48
    bar_y = 316
    bar_w = 510
    bar_h = 28
    for i, (label, ratio, color) in enumerate(progress):
        y = bar_y + i * 84
        clamped = max(0.0, min(1.0, ratio))
        parts.append(f'<text x="{bar_x}" y="{y}" fill="#c6d5ef" font-size="14" font-family="Arial">{label}</text>')
        parts.append(f'<text x="{bar_x+bar_w}" y="{y}" text-anchor="end" fill="#e8efff" font-size="14" font-family="Arial" font-weight="700">{clamped:.2%}</text>')
        parts.append(f'<rect x="{bar_x}" y="{y+12}" width="{bar_w}" height="{bar_h}" rx="10" fill="#0f1728" stroke="#2f4165" stroke-width="1"/>')
        parts.append(f'<rect x="{bar_x}" y="{y+12}" width="{bar_w * clamped:.1f}" height="{bar_h}" rx="10" fill="{color}"/>')

    parts.append(f'<text x="{bar_x}" y="792" fill="#c5d7ff" font-size="17" font-family="Arial" font-weight="700">P95 Latency: {metrics.get("p95_latency_seconds", 0.0):.2f}s  |  Requests: {int(metrics.get("request_count", 0))}</text>')

    # Right: per-agent token attribution (avg total tokens/query)
    parts.append('<rect x="602" y="254" width="548" height="556" rx="14" fill="#1f2b34" stroke="#78c6ff" stroke-width="1.2"/>')
    parts.append('<text x="620" y="289" fill="#78c6ff" font-size="22" font-family="Arial" font-weight="700">Token Distribution by Agent</text>')
    max_v = max(agent_token_breakdown.values()) if agent_token_breakdown else 1.0
    max_v = max(max_v, 1.0)
    scale_base = max_v * 1.18
    base_x = 620
    base_y = 320
    line_h = 86
    token_colors = ["#7bdff2", "#90f1b8", "#a9c1ff", "#ff9f7a", "#8ae6de", "#b4c7ff"]
    for i, (agent, value) in enumerate(agent_token_breakdown.items()):
        y = base_y + i * line_h
        ratio = max(0.0, min(1.0, float(value) / scale_base))
        bar_color = token_colors[i % len(token_colors)]
        parts.append(f'<text x="{base_x}" y="{y}" fill="#cde5f7" font-size="16" font-family="Arial">{agent}</text>')
        parts.append(f'<text x="{base_x+510}" y="{y}" text-anchor="end" fill="#ecf6ff" font-size="16" font-family="Arial" font-weight="700">{value:,.0f}</text>')
        parts.append(f'<rect x="{base_x}" y="{y+16}" width="510" height="26" rx="12" fill="#171e29" stroke="#355876" stroke-width="1"/>')
        parts.append(f'<rect x="{base_x}" y="{y+16}" width="{510 * ratio:.1f}" height="26" rx="12" fill="{bar_color}"/>')

    # Metadata cards at bottom for better readability.
    meta_y = 832
    meta_h = 44
    meta_gap = 16
    meta_w = (1120 - meta_gap) / 2
    meta_cards = [
        ("Tickers", meta.get("tickers_label", "n/a")),
        ("AI Models", meta.get("ai_models", "n/a")),
    ]
    for i, (label, value) in enumerate(meta_cards):
        x = 30 + i * (meta_w + meta_gap)
        parts.append(f'<rect x="{x:.1f}" y="{meta_y}" width="{meta_w:.1f}" height="{meta_h}" rx="10" fill="#0f1730" stroke="#304566" stroke-width="1"/>')
        parts.append(f'<text x="{x+12:.1f}" y="{meta_y+17}" fill="#8fa7d3" font-size="11" font-family="Arial">{label}</text>')
        parts.append(f'<text x="{x+12:.1f}" y="{meta_y+34}" fill="#dbe8ff" font-size="13" font-family="Arial">{value}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def _agent_token_breakdown(rows: list[dict[str, str]], metrics: dict[str, float]) -> dict[str, float]:
    # Average total tokens/query by agent groups (input+output):
    # Filing, SEC Facts, News, Writer, plus Other so totals reconcile.
    filing = 0.0
    sec_facts = 0.0
    news = 0.0
    writer = 0.0
    n = max(1, len(rows))
    for row in rows:
        writer += _to_float(row.get("prompt_tokens")) + _to_float(row.get("completion_tokens"))
        raw = row.get("agent_token_usage_json") or "{}"
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        for k, v in parsed.items():
            if not isinstance(v, dict):
                continue
            total = _to_float(v.get("total_tokens"))
            key = str(k).lower()
            if key.startswith("filings_"):
                filing += total
            elif key.startswith("sec_facts_"):
                sec_facts += total
            elif key.startswith("news_"):
                news += total
    filing_avg = round(filing / n, 2)
    sec_avg = round(sec_facts / n, 2)
    news_avg = round(news / n, 2)
    writer_avg = round(writer / n, 2)
    known_total = filing_avg + sec_avg + news_avg + writer_avg
    expected_total = float(metrics.get("avg_input_tokens", 0.0)) + float(metrics.get("avg_output_tokens", 0.0))
    other_avg = round(max(0.0, expected_total - known_total), 2)

    return {
        "Filing Agent": filing_avg,
        "SEC Facts Agent": sec_avg,
        "News Agent": news_avg,
        "Other (Planner/Critic/etc)": other_avg,
        "Writer Agent": writer_avg,
    }


def _dashboard_meta(rows: list[dict[str, str]], metrics: dict[str, float]) -> dict[str, str]:
    tickers = sorted({str(r.get("ticker", "")).strip() for r in rows if str(r.get("ticker", "")).strip()})
    models = sorted({str(r.get("writer_model", "")).strip() for r in rows if str(r.get("writer_model", "")).strip()})
    tickers_label = ", ".join(tickers) if tickers else "n/a"
    ai_models = ", ".join(models) if models else "n/a"
    return {
        "tickers_label": tickers_label,
        "ai_models": ai_models,
        "request_count": str(int(metrics.get("request_count", len(rows)))),
    }


def generate_eval_svgs(csv_path: str, metrics: dict[str, float], output_dir: str) -> dict[str, str]:
    csv_file = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_file.exists():
        return {}

    with csv_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    grouped = _group_by_ticker(rows)
    citation = {t: mean(_to_float(r.get("citation_coverage")) for r in rs) for t, rs in grouped.items()}
    latency_seconds = {
        t: mean(
            _to_float(r.get("latency_seconds"))
            if r.get("latency_seconds") not in (None, "")
            else (_to_float(r.get("latency_ms")) / 1000.0)
            for r in rs
        )
        for t, rs in grouped.items()
    }
    latency_p95_seconds = {
        t: (
            sorted(
                (
                    _to_float(r.get("latency_seconds"))
                    if r.get("latency_seconds") not in (None, "")
                    else (_to_float(r.get("latency_ms")) / 1000.0)
                )
                for r in rs
            )[max(0, int(round(0.95 * (len(rs) - 1))))]
            if rs
            else 0.0
        )
        for t, rs in grouped.items()
    }
    confidence = {t: mean(_to_float(r.get("confidence_score")) for r in rs) for t, rs in grouped.items()}
    input_tokens = {
        t: mean(
            _to_float(r.get("input_tokens"))
            if r.get("input_tokens") not in (None, "")
            else _to_float(r.get("prompt_tokens"))
            for r in rs
        )
        for t, rs in grouped.items()
    }
    output_tokens = {
        t: mean(
            _to_float(r.get("output_tokens"))
            if r.get("output_tokens") not in (None, "")
            else _to_float(r.get("completion_tokens"))
            for r in rs
        )
        for t, rs in grouped.items()
    }
    agent_breakdown = _agent_token_breakdown(rows, metrics)

    files = {
        "kpi_dashboard_option_c": out_dir / "evaluation_kpi_dashboard_option_c.svg",
        "citation_by_ticker": out_dir / "evaluation_citation_by_ticker.svg",
        "latency_by_ticker": out_dir / "evaluation_latency_by_ticker.svg",
        "confidence_by_ticker": out_dir / "evaluation_confidence_by_ticker.svg",
        "tokens_by_ticker": out_dir / "evaluation_tokens_by_ticker.svg",
        "signal_board": out_dir / "evaluation_signal_board.svg",
    }

    meta = _dashboard_meta(rows, metrics)
    _svg_kpi_dashboard_option_c(metrics, agent_breakdown, files["kpi_dashboard_option_c"], meta)
    _svg_lollipop_chart(
        "Citation Coverage by Ticker", "Analyst Deck: ranked lollipop view", citation, files["citation_by_ticker"], "#7dc8ff", is_percent=True
    )
    _svg_latency_histogram(
        "Latency by Ticker", "Histogram view: average latency (seconds)", latency_seconds, files["latency_by_ticker"]
    )
    _svg_lollipop_chart(
        "Confidence Score by Ticker", "Analyst Deck: ranked confidence meters", confidence, files["confidence_by_ticker"], "#8bf2cc", is_percent=False
    )
    _svg_stacked_tokens(
        "Tokens by Ticker", "Analyst Deck: stacked input + output tokens", input_tokens, output_tokens, files["tokens_by_ticker"]
    )
    _svg_signal_board(
        "Signal Board Summary", grouped, latency_seconds, files["signal_board"]
    )

    return {k: str(v) for k, v in files.items()}
