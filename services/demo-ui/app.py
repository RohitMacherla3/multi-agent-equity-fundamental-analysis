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
import streamlit as st

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None
    letter = None


st.set_page_config(page_title="Multi-Agent Equity Fundamental Analysis", page_icon="📈", layout="wide")
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

JAVA_APP_URL = st.sidebar.text_input("Java service URL", os.getenv("JAVA_APP_URL", "http://localhost:8080"))
INDEX_APP_URL = st.sidebar.text_input("Indexing service URL", os.getenv("INDEX_APP_URL", "http://localhost:8002"))
PY_APP_URL = st.sidebar.text_input("Python service URL", os.getenv("PY_APP_URL", "http://localhost:8000"))
TIMEOUT_SEC = st.sidebar.number_input("HTTP timeout (sec)", min_value=2, max_value=120, value=20)
ANALYSIS_TIMEOUT_SEC = st.sidebar.number_input("Agent analysis timeout (sec)", min_value=10, max_value=600, value=180)
DEBUG_MODE = st.sidebar.checkbox(
    "Debug mode",
    value=os.getenv("AGENT_DEBUG_DEFAULT", "false").lower() == "true",
    help="Show raw technical trace and pass-level internals.",
)
THIS_FILE = Path(__file__).resolve()


def get_json(url: str) -> tuple[bool, Any]:
    try:
        with httpx.Client(timeout=float(TIMEOUT_SEC)) as client:
            r = client.get(url)
        r.raise_for_status()
        return True, r.json()
    except Exception as exc:
        return False, str(exc)


def post_json(url: str, payload: dict[str, Any], timeout_sec: float | None = None) -> tuple[bool, Any]:
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
    ("SEC Facts Agent", {"sec_facts_agent_pass1", "sec_facts_agent_pass2", "sec_facts_agent"}),
    ("News Agent", {"news_agent_pass1", "news_agent_pass2", "news_agent"}),
    ("Research Aggregator", {"research_aggregator_pass1", "research_aggregator_pass2", "research_aggregator", "research_swarm_pass1", "research_swarm_pass2"}),
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


def _first_matching_step(trace: list[dict[str, Any]], keys: set[str]) -> dict[str, Any] | None:
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


def _fetch_existing_filings_for_ticker(ticker: str, cik: str, limit: int = 200) -> list[dict[str, Any]]:
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


def _draw_wrapped_text(pdf: Any, text: str, x: int, y: int, max_width: int, line_height: int = 14) -> int:
    words = text.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if pdf.stringWidth(trial, "Helvetica", 10) <= max_width:
            line = trial
        else:
            pdf.drawString(x, y, line)
            y -= line_height
            line = word
            if y < 60:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = 740
    if line:
        pdf.drawString(x, y, line)
        y -= line_height
    return y


def _ensure_page_space(pdf: Any, y: int, min_y: int = 80) -> int:
    if y < min_y:
        pdf.showPage()
        pdf.setFont("Helvetica", 10)
        return 740
    return y


def _draw_multiline_text(
    pdf: Any,
    text: str,
    x: int,
    y: int,
    max_width: int,
    line_height: int = 14,
) -> int:
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            y -= 6
            y = _ensure_page_space(pdf, y)
            continue
        if line.startswith("- "):
            y = _draw_wrapped_text(pdf, line, x, y, max_width, line_height=line_height)
        else:
            y = _draw_wrapped_text(pdf, line, x, y, max_width, line_height=line_height)
        y = _ensure_page_space(pdf, y)
    return y


def build_pdf_report(
    analysis_response: dict[str, Any],
    query: str,
    ticker: str | None,
    trace: list[dict[str, Any]],
) -> bytes | None:
    if canvas is None:
        return None

    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Multi-Agent Equity Fundamental Analysis - Report")
    y -= 20
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()}Z")
    y -= 14
    pdf.drawString(50, y, f"Ticker: {ticker or 'N/A'}")
    y -= 14
    pdf.drawString(50, y, f"Query: {query}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Report")
    y -= 16
    pdf.setFont("Helvetica", 10)
    memo_text = str(analysis_response.get("memo_markdown", "")).replace("#", "").replace("`", "")
    y = _draw_multiline_text(pdf, memo_text, 50, y, max_width=520)
    y -= 8
    y = _ensure_page_space(pdf, y)
    pdf.drawString(50, y, f"Confidence: {analysis_response.get('confidence', 'N/A')}")
    y -= 14
    pdf.drawString(50, y, f"Writer model: {analysis_response.get('writer_model', 'unknown')}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Agent Results")
    y -= 16
    pdf.setFont("Helvetica", 10)
    for label, keys in AGENT_FLOW:
        step = _first_matching_step(trace, keys)
        if not step:
            continue
        output = step.get("output", {}) if isinstance(step, dict) else {}
        summary = str(output.get("summary_text") or output.get("summary") or output.get("candidate_summary") or "")
        if not summary:
            continue
        count = output.get("count")
        provider = output.get("provider")
        top_items = output.get("top_items", []) if isinstance(output.get("top_items", []), list) else []

        y = _ensure_page_space(pdf, y, min_y=120)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, y, label)
        y -= 14
        pdf.setFont("Helvetica", 10)
        meta_line = []
        if isinstance(count, int):
            meta_line.append(f"evidence={count}")
        if provider:
            meta_line.append(f"provider={provider}")
        if meta_line:
            y = _draw_wrapped_text(pdf, " | ".join(meta_line), 50, y, max_width=520)

        y = _draw_multiline_text(pdf, summary, 50, y, max_width=520)
        for idx, item in enumerate(top_items[:2], start=1):
            ref = (
                f"[{idx}] {item.get('ticker','')} {item.get('form_type','')} {item.get('section_name','')}"
            ).strip()
            y = _draw_wrapped_text(pdf, f"Reference {idx}: {ref}", 50, y, max_width=520)
        y -= 10
        y = _ensure_page_space(pdf, y)

    y -= 8
    y = _ensure_page_space(pdf, y, min_y=120)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Evidence & Citations")
    y -= 16
    pdf.setFont("Helvetica", 10)
    evidence = analysis_response.get("evidence", []) or []
    for idx, ev in enumerate(evidence, start=1):
        y = _ensure_page_space(pdf, y, min_y=110)
        line = (
            f"[{idx}] {ev.get('ticker','')} {ev.get('form_type','')} {ev.get('section_name','')} "
            f"accession={ev.get('accession_no','')} score={ev.get('score','')}"
        )
        y = _draw_wrapped_text(pdf, line, 50, y, max_width=520)
        preview = str(ev.get("text_preview", ""))
        if preview:
            y = _draw_wrapped_text(pdf, f"    {preview}", 50, y, max_width=520)
        y -= 6

    pdf.save()
    buf.seek(0)
    return buf.read()


def render_agent_insights(trace: list[dict[str, Any]], debug_mode: bool = False) -> None:
    st.markdown("### Agent Insights")
    lookup = _trace_lookup(trace)
    cards_rendered = 0
    for label, keys in AGENT_FLOW:
        step = _first_matching_step(trace, keys)
        if not step:
            continue
        output = step.get("output", {}) if isinstance(step, dict) else {}
        summary_text = str(output.get("summary_text") or output.get("candidate_summary") or output.get("summary") or "").strip()
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


tab_ingest_index, tab_agent = st.tabs([
    "Ingestion + Indexing",
    "Agentic Analysis",
])

with tab_ingest_index:
    st.subheader("Process One Ticker")
    st.write("Select a ticker and click Process. The app will ingest filings, then index them into Chroma.")

    selected = st.selectbox("Ticker", options=list(TICKER_CHOICES.keys()), index=0)
    ticker, cik = TICKER_CHOICES[selected]
    filings_to_index = st.slider("Filings to index", min_value=1, max_value=10, value=10)

    if st.button("Process Ticker", type="primary", width="stretch"):
        with st.spinner(f"Running ingestion + indexing for {ticker}..."):
            ok_java = wait_for_service_health(f"{JAVA_APP_URL}/actuator/health", timeout_sec=120)
            ok_idx = wait_for_service_health(f"{INDEX_APP_URL}/health", timeout_sec=120)
            ok_ai = wait_for_service_health(f"{PY_APP_URL}/health", timeout_sec=120)
            if not (ok_java and ok_idx and ok_ai):
                st.error(
                    "Services are not healthy. Run `make all-services` from project root, "
                    "then retry."
                )
                st.stop()

            existing_filings = _fetch_existing_filings_for_ticker(ticker, cik, limit=200)
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
                ok_start, start_resp = post_json(f"{JAVA_APP_URL}/v1/ingestion/run", ingest_payload)
                if not ok_start:
                    st.error(f"Ingestion trigger failed: {start_resp}")
                    st.stop()

                run_id = start_resp.get("runId", "")
                if not run_id:
                    st.error("Ingestion trigger response did not return runId")
                    st.stop()

                ok_wait, run_status = wait_for_ingestion_run(run_id=run_id, timeout_sec=180)
                if not ok_wait:
                    st.error(f"Ingestion polling failed: {run_status}")
                    st.stop()

                final_status = str(run_status.get("status", "UNKNOWN")).upper()
                if final_status not in {"SUCCEEDED", "PARTIAL_SUCCESS"}:
                    st.error(f"Ingestion run ended with status {final_status}")
                    with st.expander("Run status details"):
                        show_json(run_status)
                    st.stop()

                filings = _fetch_existing_filings_for_ticker(ticker, cik, limit=200)[: int(filings_to_index)]
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
            ok_index, index_resp = post_json(f"{INDEX_APP_URL}/v1/indexing/index-batch", index_payload)
            if not ok_index:
                st.error(f"Indexing failed: {index_resp}")
                st.stop()

            ok_stats, stats_resp = get_json(f"{INDEX_APP_URL}/v1/indexing/stats")
            chunk_count = stats_resp.get("chunkCount", "N/A") if ok_stats and isinstance(stats_resp, dict) else "N/A"

            st.session_state["processed_ticker"] = ticker
            st.session_state["processed_cik"] = cik
            indexed_tickers = [ticker]
            st.session_state["indexed_tickers"] = indexed_tickers
            st.session_state["pipeline_ready"] = True
            st.session_state["last_ingestion_run"] = run_status
            st.session_state["last_indexing_result"] = index_resp
            st.session_state["last_reset_result"] = reset_resp if ok_reset else {"warning": str(reset_resp)}

            st.success(f"Completed for {ticker}. Indexed={index_resp.get('indexed', 0)} Failed={index_resp.get('failed', 0)} | Chunks={chunk_count}")
            st.info("Next step: open the Agentic Analysis tab.")

    if st.session_state.get("pipeline_ready"):
        st.write("Current prepared ticker:", st.session_state.get("processed_ticker", ""))
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
    current_engine = py_health.get("agentEngine", "unknown") if ok_health and isinstance(py_health, dict) else "unknown"
    st.caption(f"Current agent engine: `{current_engine}`")
    if current_engine != "langgraph":
        st.warning("LangGraph swarm is not active. Restart Python AI service from Health tab using the Start button.")
    default_ticker = st.session_state.get("processed_ticker", "")
    agent_query = st.text_area("Analysis query", value="What are the key risk and growth signals in the latest filings?")
    agent_ticker = st.text_input("Ticker (optional)", value=default_ticker)
    agent_top_k = st.slider("Agent topK", min_value=1, max_value=20, value=8)
    agent_max_evidence = st.slider("Max evidence", min_value=1, max_value=25, value=12)

    if not st.session_state.get("pipeline_ready"):
        st.warning("Run Ingestion + Indexing first, then come back here.")
    elif st.session_state.get("indexed_tickers"):
        st.caption(f"Indexed tickers in current vector set: {', '.join(st.session_state.get('indexed_tickers', []))}")

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
                st.caption("PDF export requires `reportlab`. Install with: `pip install reportlab`")
        else:
            resp_text = str(resp)
            if "timed out" in resp_text.lower():
                st.error(
                    f"Request timed out after {int(ANALYSIS_TIMEOUT_SEC)}s. "
                    "Increase 'Agent analysis timeout (sec)' in the sidebar or retry with fewer evidence chunks."
                )
            else:
                st.error(resp)
