from __future__ import annotations

import json
import os
import time
import re
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import sys
import streamlit as st

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
        PageBreak,
        KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

    _reportlab_ok = True
except Exception:
    canvas = None
    letter = None
    _reportlab_ok = False

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

st.set_page_config(
    page_title="Multi-Agent Equity Fundamental Analysis", page_icon="📈", layout="wide"
)
st.title("Multi-Agent Equity Fundamental Analysis")


TICKER_CHOICES = {
    "NVDA - NVIDIA": ("NVDA", "0001045810"),
    "MSFT - Microsoft": ("MSFT", "0000789019"),
    "AAPL - Apple": ("AAPL", "0000320193"),
    "AMZN - Amazon": ("AMZN", "0001018724"),
    "GOOGL - Alphabet": ("GOOGL", "0001652044"),
    "META - Meta": ("META", "0001326801"),
    "TSLA - Tesla": ("TSLA", "0001318605"),
    "JPM - JPMorgan": ("JPM", "0000019617"),
    "GS - Goldman Sachs": ("GS", "0000886982"),
    "BRK-B - Berkshire Hathaway": ("BRK-B", "0001067983"),
}

JAVA_APP_URL = st.sidebar.text_input(
    "Java service URL", os.getenv("JAVA_APP_URL", "http://localhost:8080")
)
INDEX_APP_URL = st.sidebar.text_input(
    "Indexing service URL", os.getenv("INDEX_APP_URL", "http://localhost:8002")
)
PY_APP_URL = st.sidebar.text_input(
    "Python service URL", os.getenv("PY_APP_URL", "http://localhost:8000")
)
TIMEOUT_SEC = st.sidebar.number_input(
    "HTTP timeout (sec)", min_value=2, max_value=120, value=20
)
ANALYSIS_TIMEOUT_SEC = st.sidebar.number_input(
    "Agent analysis timeout (sec)", min_value=10, max_value=600, value=180
)
DEBUG_MODE = st.sidebar.checkbox(
    "Debug mode",
    value=os.getenv("AGENT_DEBUG_DEFAULT", "false").lower() == "true",
    help="Show raw technical trace and pass-level internals.",
)
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]  # multi-agent-equity-fundamental-analysis/
RESULTS_DIR = REPO_ROOT / "data" / "results"


def get_json(url: str) -> tuple[bool, Any]:
    try:
        with httpx.Client(timeout=float(TIMEOUT_SEC)) as client:
            r = client.get(url)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:
        return False, str(exc)


def post_json(
    url: str, payload: dict[str, Any], timeout_sec: float | None = None
) -> tuple[bool, Any]:
    try:
        timeout = float(timeout_sec) if timeout_sec is not None else float(TIMEOUT_SEC)
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:
        return False, str(exc)


def show_json(data: Any) -> None:
    st.code(json.dumps(data, indent=2), language="json")


def normalize_ticker(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def wait_for_ingestion_run(run_id: str, timeout_sec: int = 120) -> tuple[bool, Any]:
    deadline = time.time() + timeout_sec
    last_payload: Any = None
    while time.time() < deadline:
        ok, payload = get_json(f"{JAVA_APP_URL}/v1/ingestion/runs/{run_id}")
        if not ok:
            return False, payload
        last_payload = payload
        status = str(payload.get("status", "")).upper()
        if status in {"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED"}:
            return True, payload
        time.sleep(2)
    return False, last_payload or "Timed out waiting for ingestion run status"


def wait_for_service_health(url: str, timeout_sec: int = 120) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        ok, _ = get_json(url)
        if ok:
            return True
        time.sleep(2)
    return False


AGENT_FLOW = [
    ("Filing Agent", {"filing_agent_pass1", "filing_agent_pass2", "filing_agent"}),
    (
        "SEC Facts Agent",
        {"sec_facts_agent_pass1", "sec_facts_agent_pass2", "sec_facts_agent"},
    ),
    ("News Agent", {"news_agent_pass1", "news_agent_pass2", "news_agent"}),
    (
        "Research Aggregator",
        {
            "research_aggregator_pass1",
            "research_aggregator_pass2",
            "research_aggregator",
            "research_swarm_pass1",
            "research_swarm_pass2",
        },
    ),
    ("Critic Agent", {"critic_eval_pass1", "critic_eval_pass2", "critic"}),
    ("Comparison Agent", {"comparison_agent"}),
    ("Writer Agent", {"writer"}),
]


def _trace_lookup(trace: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for step in trace:
        name = str(step.get("agent", "")).strip()
        if name:
            out[name] = step
    return out


def _first_matching_step(
    trace: list[dict[str, Any]], keys: set[str]
) -> dict[str, Any] | None:
    for step in trace:
        if str(step.get("agent", "")).strip() in keys:
            return step
    return None


def _render_top_item_line(item: dict[str, Any], idx: int) -> str:
    label = (
        f"[{idx}] {item.get('ticker', '')} {item.get('form_type', '')} "
        f"{item.get('section_name', '')}"
    ).strip()
    url = str(item.get("source_url") or "").strip()
    if url:
        return f"- [{label}]({url})"
    return f"- {label}"


def _fetch_existing_filings_for_ticker(
    ticker: str, cik: str, limit: int = 200
) -> list[dict[str, Any]]:
    ok, payload = get_json(f"{JAVA_APP_URL}/v1/filings?limit={limit}")
    if not ok or not isinstance(payload, list):
        return []
    target_ticker = normalize_ticker(ticker)
    out = [
        row
        for row in payload
        if str(row.get("cik", "")).strip() == str(cik).strip()
        or normalize_ticker(str(row.get("ticker", ""))) == target_ticker
    ]
    return out


# ── PDF Helpers ───────────────────────────────────────────────────────────────

MONEY_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|thousand|B|M|T|K))?",
    re.IGNORECASE,
)


def _safe_xml(text: str) -> str:
    """Escape XML special chars for use in Platypus Paragraph markup."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _highlight_money(text: str) -> str:
    """Escape XML then wrap monetary amounts in amber bold markup for Platypus."""
    escaped = _safe_xml(text)
    return MONEY_RE.sub(
        lambda m: f'<font color="#047857"><b>{m.group()}</b></font>',
        escaped,
    )


def _parse_memo_sections(memo_markdown: str) -> dict[str, str]:
    """Split a ## markdown memo into a dict keyed by section headings."""
    sections: dict[str, list[str]] = {"_intro": []}
    current = "_intro"
    for line in memo_markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif line.startswith("# "):
            current = line[2:].strip()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def build_pdf_report(
    analysis_response: dict[str, Any],
    query: str,
    ticker: str | None,
    trace: list[dict[str, Any]],
) -> bytes | None:
    """Build a richly-formatted PDF memo, save it to data/results/<ticker>/, and return bytes."""
    if not _reportlab_ok:
        return None

    buf = BytesIO()
    safe_ticker = (ticker or "unknown").replace("/", "-")
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"MEFA — {safe_ticker}",
        author="Multi-Agent Equity Fundamental Analysis",
    )
    styles = getSampleStyleSheet()

    # ── Custom styles ──────────────────────────────────────────────────────────
    navy = rl_colors.HexColor("#1e3a5f")
    green_dark = rl_colors.HexColor("#065f46")
    red_dark = rl_colors.HexColor("#7f1d1d")
    purple_dark = rl_colors.HexColor("#44337a")
    slate = rl_colors.HexColor("#374151")
    bg_light = rl_colors.HexColor("#f0f4f8")

    title_style = ParagraphStyle(
        "RPTitle", parent=styles["Title"], fontSize=18, textColor=navy, spaceAfter=4
    )
    h1 = ParagraphStyle(
        "RPH1",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=navy,
        spaceBefore=12,
        spaceAfter=4,
    )
    h2 = ParagraphStyle(
        "RPH2",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=slate,
        spaceBefore=8,
        spaceAfter=3,
    )
    body = ParagraphStyle(
        "RPBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=3,
        alignment=TA_JUSTIFY,
    )
    bullet_s = ParagraphStyle(
        "RPBullet",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        leftIndent=14,
        spaceAfter=2,
    )
    meta_s = ParagraphStyle(
        "RPMeta", parent=styles["Normal"], fontSize=9, textColor=slate, spaceAfter=2
    )
    cite_s = ParagraphStyle(
        "RPCite",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        leftIndent=10,
        spaceAfter=2,
        textColor=rl_colors.HexColor("#333333"),
    )
    hdr_cite_s = ParagraphStyle(
        "RPHdrCite",
        parent=cite_s,
        textColor=rl_colors.white,
        leftIndent=0,
    )
    footer_s = ParagraphStyle(
        "RPFooter",
        parent=styles["Normal"],
        fontSize=7,
        textColor=rl_colors.grey,
        alignment=TA_CENTER,
    )

    SECTION_COLORS = {
        "Executive Summary": navy,
        "Bull Case Signals": green_dark,
        "Risk Signals": red_dark,
        "What To Verify Next": purple_dark,
    }

    def _p(text: str, style: Any = body) -> Paragraph:
        return Paragraph(_highlight_money(text), style)

    def _bullet(text: str) -> Paragraph:
        return Paragraph(f"\u2022\u00a0{_highlight_money(text)}", bullet_s)

    def _hr(color: Any = navy, thickness: float = 1.0) -> HRFlowable:
        return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4)

    # ── Story ──────────────────────────────────────────────────────────────────
    story: list[Any] = []

    # Title + timestamp
    story.append(Paragraph("Multi-Agent Equity Fundamental Analysis", title_style))
    story.append(_hr(navy, 2))

    # Metadata row
    confidence = analysis_response.get("confidence", "N/A")
    writer_model = analysis_response.get("writer_model", "unknown")
    ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    meta_rows = [
        [
            Paragraph(f"<b>Ticker:</b>\u00a0{safe_ticker}", meta_s),
            Paragraph(f"<b>Confidence:</b>\u00a0{confidence}", meta_s),
            Paragraph(f"<b>Model:</b>\u00a0{writer_model}", meta_s),
            Paragraph(f"<b>Date:</b>\u00a0{ts_str}", meta_s),
        ]
    ]
    meta_tbl = Table(meta_rows, colWidths=["25%", "25%", "25%", "25%"])
    meta_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_light),
                ("BOX", (0, 0), (-1, -1), 0.5, rl_colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, rl_colors.lightgrey),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(meta_tbl)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Query:</b>\u00a0{_safe_xml(query)}", meta_s))
    story.append(Spacer(1, 10))

    # ── Memo sections ──────────────────────────────────────────────────────────
    memo_md = analysis_response.get("memo_markdown", "")
    sections = _parse_memo_sections(memo_md)
    SECTION_ORDER = [
        "Executive Summary",
        "Bull Case Signals",
        "Risk Signals",
        "What To Verify Next",
    ]

    def _render_section(sec_name: str, content: str, color: Any = navy) -> None:
        story.append(
            Paragraph(
                f"<b>{_safe_xml(sec_name)}</b>",
                ParagraphStyle(f"_h1_{sec_name}", parent=h1, textColor=color),
            )
        )
        story.append(_hr(color, 0.6))
        for line in content.splitlines():
            line = line.strip()
            if not line:
                story.append(Spacer(1, 3))
                continue
            if line.startswith(("- ", "* ")):
                story.append(_bullet(line[2:]))
            elif line.startswith("**") and line.endswith("**"):
                story.append(_p(f"<b>{_safe_xml(line[2:-2])}</b>", body))
            else:
                story.append(_p(line, body))
        story.append(Spacer(1, 8))

    for sec_name in SECTION_ORDER:
        content = sections.get(sec_name, "")
        if content:
            _render_section(sec_name, content, SECTION_COLORS.get(sec_name, navy))

    # Any remaining sections not in the preset order
    for sec_name, content in sections.items():
        if (
            sec_name in SECTION_ORDER
            or sec_name in ("_intro", "", "Citations")
            or not content.strip()
        ):
            continue
        _render_section(sec_name, content)

    # ── Evidence table ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Evidence", h1))
    story.append(_hr())
    evidence = analysis_response.get("evidence", []) or []
    if evidence:
        ev_hdr = [
            Paragraph("<b>#</b>", hdr_cite_s),
            Paragraph("<b>Ticker / Form</b>", hdr_cite_s),
            Paragraph("<b>Section</b>", hdr_cite_s),
            Paragraph("<b>Score</b>", hdr_cite_s),
            Paragraph("<b>Preview</b>", hdr_cite_s),
        ]
        ev_data = [ev_hdr]
        for idx, ev in enumerate(evidence, 1):
            score = ev.get("score", "")
            score_str = f"{float(score):.3f}" if score != "" else "—"
            preview = str(ev.get("text_preview", ""))[:220]
            src_url = str(ev.get("source_url") or "").strip()
            acc = _safe_xml(str(ev.get("accession_no", "")))
            acc_para = (
                Paragraph(f'<a href="{src_url}" color="#1e3a5f">{acc}</a>', cite_s)
                if src_url
                else Paragraph(acc, cite_s)
            )
            ev_data.append(
                [
                    Paragraph(str(idx), cite_s),
                    acc_para,
                    Paragraph(
                        _safe_xml(f"{ev.get('ticker', '')} {ev.get('form_type', '')}"),
                        cite_s,
                    ),
                    Paragraph(score_str, cite_s),
                    Paragraph(_highlight_money(preview), cite_s),
                ]
            )
        ev_tbl = Table(
            ev_data, colWidths=[0.3 * inch, 1.5 * inch, 1.0 * inch, 0.55 * inch, None]
        )
        ev_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, bg_light]),
                    ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.lightgrey),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (3, 0), (3, -1), "CENTER"),
                ]
            )
        )
        story.append(ev_tbl)
    else:
        story.append(_p("No evidence retrieved.", body))
    story.append(Spacer(1, 10))

    # ── Citations ──────────────────────────────────────────────────────────────
    cite_content = sections.get("Citations", "")
    if cite_content:
        story.append(Paragraph("Citations", h1))
        story.append(_hr())
        for line in cite_content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Linkify source_url=... patterns that appear in citation text
            url_m = re.search(r"(?:source_url=|url=)(\S+)", line)
            if url_m:
                url = url_m.group(1)
                linked = _safe_xml(line).replace(
                    _safe_xml(url),
                    f'<a href="{url}" color="#1e3a5f">{_safe_xml(url)}</a>',
                )
                story.append(Paragraph(linked, cite_s))
            else:
                story.append(Paragraph(_highlight_money(line), cite_s))
        story.append(Spacer(1, 8))

    # ── Agent Insights ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Agent Insights", h1))
    story.append(_hr())
    for label, keys in AGENT_FLOW:
        step = _first_matching_step(trace, keys)
        if not step:
            continue
        output = step.get("output", {}) if isinstance(step, dict) else {}
        summary_text = str(
            output.get("summary_text")
            or output.get("candidate_summary")
            or output.get("summary")
            or ""
        ).strip()
        if not summary_text:
            continue
        count = output.get("count")
        provider = output.get("provider")
        top_items = output.get("top_items") or []

        meta_parts = [f"<b>{_safe_xml(label)}</b>"]
        if isinstance(count, int):
            meta_parts.append(f"evidence: {count}")
        if provider:
            meta_parts.append(f"provider: <i>{provider}</i>")
        story.append(Paragraph(" \u00b7 ".join(meta_parts), h2))
        for ln in summary_text.splitlines():
            ln = ln.strip()
            if not ln:
                story.append(Spacer(1, 3))
            elif ln.startswith(("- ", "* ")):
                story.append(_bullet(ln[2:]))
            else:
                story.append(_p(ln, body))
        for i, item in enumerate(top_items[:3], 1):
            url = str(item.get("source_url") or "").strip()
            ref = _safe_xml(
                f"{item.get('ticker', '')} {item.get('form_type', '')} {item.get('section_name', '')}".strip()
            )
            if url:
                story.append(
                    Paragraph(
                        f'[{i}]\u00a0<a href="{url}" color="#1e3a5f">{ref}</a>', cite_s
                    )
                )
            else:
                story.append(Paragraph(f"[{i}]\u00a0{ref}", cite_s))
        story.append(Spacer(1, 6))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(_hr(rl_colors.lightgrey, 0.5))
    story.append(
        Paragraph(
            "Generated by Multi-Agent Equity Fundamental Analysis. "
            "For research purposes only — not investment advice.",
            footer_s,
        )
    )

    doc.build(story)
    pdf_bytes = buf.getvalue()

    # Persist to disk
    save_dir = RESULTS_DIR / safe_ticker
    save_dir.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"mefa_{safe_ticker}_{ts_file}.pdf"
    save_path.write_bytes(pdf_bytes)

    return pdf_bytes


def render_agent_insights(
    trace: list[dict[str, Any]], debug_mode: bool = False
) -> None:
    st.markdown("### Agent Insights")
    lookup = _trace_lookup(trace)
    cards_rendered = 0
    for label, keys in AGENT_FLOW:
        step = _first_matching_step(trace, keys)
        if not step:
            continue
        output = step.get("output", {}) if isinstance(step, dict) else {}
        summary_text = str(
            output.get("summary_text")
            or output.get("candidate_summary")
            or output.get("summary")
            or ""
        ).strip()
        count = output.get("count")
        provider = output.get("provider")
        top_items = output.get("top_items", [])

        with st.container(border=True):
            st.markdown(f"✅ **{label}**")
            if provider:
                st.caption(f"Provider: `{provider}`")
            if isinstance(count, int):
                st.caption(f"Evidence count: `{count}`")
            if summary_text:
                st.write(summary_text)
            if top_items:
                st.markdown("Top references:")
                for i, item in enumerate(top_items[:2], start=1):
                    st.markdown(_render_top_item_line(item, i))
        cards_rendered += 1

    if cards_rendered == 0:
        st.info("No agent trace available for this run.")

    if debug_mode and trace:
        st.markdown("### Technical Trace (Debug)")
        st.json(trace)


tab_ingest_index, tab_agent = st.tabs(
    [
        "Ingestion + Indexing",
        "Agentic Analysis",
    ]
)

with tab_ingest_index:
    st.subheader("Process One Ticker")
    st.write(
        "Select a ticker and click Process. The app will ingest filings, then index them into Chroma."
    )

    selected = st.selectbox("Ticker", options=list(TICKER_CHOICES.keys()), index=0)
    ticker, cik = TICKER_CHOICES[selected]
    filings_to_index = st.slider(
        "Filings to index", min_value=1, max_value=10, value=10
    )

    if st.button("Process Ticker", type="primary", width="stretch"):
        with st.spinner(f"Running ingestion + indexing for {ticker}..."):
            ok_java = wait_for_service_health(
                f"{JAVA_APP_URL}/actuator/health", timeout_sec=120
            )
            ok_idx = wait_for_service_health(f"{INDEX_APP_URL}/health", timeout_sec=120)
            ok_ai = wait_for_service_health(f"{PY_APP_URL}/health", timeout_sec=120)
            if not (ok_java and ok_idx and ok_ai):
                st.error(
                    "Services are not healthy. Run `make all-services` from project root, "
                    "then retry."
                )
                st.stop()

            existing_filings = _fetch_existing_filings_for_ticker(
                ticker, cik, limit=200
            )
            run_status: dict[str, Any]
            if len(existing_filings) >= int(filings_to_index):
                run_status = {
                    "status": "SKIPPED_EXISTING",
                    "message": f"Found {len(existing_filings)} existing filings for {ticker}; skipped ingestion.",
                }
                filings = existing_filings[: int(filings_to_index)]
            else:
                ingest_payload = {
                    "ciks": [cik],
                    "maxPerCik": int(filings_to_index),
                    "includeDocuments": False,
                }
                ok_start, start_resp = post_json(
                    f"{JAVA_APP_URL}/v1/ingestion/run", ingest_payload
                )
                if not ok_start:
                    st.error(f"Ingestion trigger failed: {start_resp}")
                    st.stop()

                run_id = start_resp.get("runId", "")
                if not run_id:
                    st.error("Ingestion trigger response did not return runId")
                    st.stop()

                ok_wait, run_status = wait_for_ingestion_run(
                    run_id=run_id, timeout_sec=180
                )
                if not ok_wait:
                    st.error(f"Ingestion polling failed: {run_status}")
                    st.stop()

                final_status = str(run_status.get("status", "UNKNOWN")).upper()
                if final_status not in {"SUCCEEDED", "PARTIAL_SUCCESS"}:
                    st.error(f"Ingestion run ended with status {final_status}")
                    with st.expander("Run status details"):
                        show_json(run_status)
                    st.stop()

                filings = _fetch_existing_filings_for_ticker(ticker, cik, limit=200)[
                    : int(filings_to_index)
                ]
            if not filings:
                st.error(f"No filings found for {ticker} after ingestion")
                st.stop()

            index_payload = {
                "filings": [
                    {
                        "accessionNo": x.get("accessionNo"),
                        "sourceUrl": x.get("sourceUrl"),
                        "ticker": ticker,
                        "companyName": x.get("companyName"),
                        "formType": x.get("formType"),
                        "filingDate": x.get("filingDate"),
                    }
                    for x in filings
                ]
            }
            ok_reset, reset_resp = post_json(f"{INDEX_APP_URL}/v1/indexing/reset", {})
            if not ok_reset:
                st.warning(f"Could not reset index before indexing: {reset_resp}")
            ok_index, index_resp = post_json(
                f"{INDEX_APP_URL}/v1/indexing/index-batch",
                index_payload,
                timeout_sec=float(ANALYSIS_TIMEOUT_SEC),
            )
            if not ok_index:
                st.error(f"Indexing failed: {index_resp}")
                st.stop()

            ok_stats, stats_resp = get_json(f"{INDEX_APP_URL}/v1/indexing/stats")
            chunk_count = (
                stats_resp.get("chunkCount", "N/A")
                if ok_stats and isinstance(stats_resp, dict)
                else "N/A"
            )

            st.session_state["processed_ticker"] = ticker
            st.session_state["processed_cik"] = cik
            indexed_tickers = [ticker]
            st.session_state["indexed_tickers"] = indexed_tickers
            st.session_state["pipeline_ready"] = True
            st.session_state["last_ingestion_run"] = run_status
            st.session_state["last_indexing_result"] = index_resp
            st.session_state["last_reset_result"] = (
                reset_resp if ok_reset else {"warning": str(reset_resp)}
            )

            st.success(
                f"Completed for {ticker}. Indexed={index_resp.get('indexed', 0)} Failed={index_resp.get('failed', 0)} | Chunks={chunk_count}"
            )
            st.info("Next step: open the Agentic Analysis tab.")

    if st.session_state.get("pipeline_ready"):
        st.write(
            "Current prepared ticker:", st.session_state.get("processed_ticker", "")
        )
        with st.expander("Latest pipeline details"):
            st.markdown("**Ingestion run**")
            show_json(st.session_state.get("last_ingestion_run", {}))
            st.markdown("**Index reset**")
            show_json(st.session_state.get("last_reset_result", {}))
            st.markdown("**Indexing result**")
            show_json(st.session_state.get("last_indexing_result", {}))

with tab_agent:
    st.subheader("Agent Research Team Analysis")
    ok_health, py_health = get_json(f"{PY_APP_URL}/health")
    current_engine = (
        py_health.get("agentEngine", "unknown")
        if ok_health and isinstance(py_health, dict)
        else "unknown"
    )
    st.caption(f"Current agent engine: `{current_engine}`")
    if current_engine != "langgraph":
        st.warning(
            "LangGraph swarm is not active. Restart Python AI service from Health tab using the Start button."
        )
    default_ticker = st.session_state.get("processed_ticker", "")
    agent_query = st.text_area(
        "Analysis query",
        value="What are the key risk and growth signals in the latest filings?",
    )
    agent_ticker = st.text_input("Ticker (optional)", value=default_ticker)
    agent_top_k = st.slider("Agent topK", min_value=1, max_value=20, value=8)
    agent_max_evidence = st.slider("Max evidence", min_value=1, max_value=25, value=12)

    if not st.session_state.get("pipeline_ready"):
        st.warning("Run Ingestion + Indexing first, then come back here.")
    elif st.session_state.get("indexed_tickers"):
        st.caption(
            f"Indexed tickers in current vector set: {', '.join(st.session_state.get('indexed_tickers', []))}"
        )

    if st.button("Run agent analysis", type="primary", width="stretch"):
        payload = {
            "query": agent_query,
            "ticker": agent_ticker.strip() or None,
            "topK": int(agent_top_k),
            "maxEvidence": int(agent_max_evidence),
            "includeTrace": True,
        }
        run_status_placeholder = st.empty()
        run_status_placeholder.info("Running agent workflow...")
        with st.spinner("Executing multi-agent analysis..."):
            ok, resp = post_json(
                f"{PY_APP_URL}/v1/agents/analyze",
                payload,
                timeout_sec=float(ANALYSIS_TIMEOUT_SEC),
            )
        if ok:
            run_status_placeholder.success("Agent workflow completed")
        else:
            run_status_placeholder.error("Agent workflow failed")

        if ok:
            st.markdown("### Executive Summary")
            st.write(resp.get("summary", ""))
            st.write(f"Confidence: **{resp.get('confidence', 'N/A')}**")
            st.write(f"Writer model: `{resp.get('writer_model', 'unknown')}`")

            evidence = resp.get("evidence", [])
            if evidence:
                st.markdown("### Evidence")
                st.dataframe(evidence, width="stretch")

            st.markdown("### Memo")
            st.markdown(resp.get("memo_markdown", ""))

            trace = resp.get("trace", [])
            render_agent_insights(trace, debug_mode=DEBUG_MODE)

            pdf_bytes = build_pdf_report(
                analysis_response=resp,
                query=agent_query,
                ticker=agent_ticker.strip() or default_ticker,
                trace=trace,
            )
            if pdf_bytes is not None:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"mefa_analysis_{(agent_ticker.strip() or default_ticker or 'report').lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
            else:
                st.caption(
                    "PDF export requires `reportlab`. Install with: `pip install reportlab`"
                )
        else:
            resp_text = str(resp)
            if "timed out" in resp_text.lower():
                st.error(
                    f"Request timed out after {int(ANALYSIS_TIMEOUT_SEC)}s. "
                    "Increase 'Agent analysis timeout (sec)' in the sidebar or retry with fewer evidence chunks."
                )
            else:
                st.error(resp)
