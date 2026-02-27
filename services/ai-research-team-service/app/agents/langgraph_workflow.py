from __future__ import annotations

import json
import logging
import re
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypedDict

import httpx
from pydantic import BaseModel, Field

from app.agents.news_clients import NewsFetchResult, TavilyNewsClient
from app.core.config import settings
from app.retrieval.indexer import indexer

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None
    StateGraph = None
    START = None
    END = None


class ResearchPlan(BaseModel):
    filing_queries: list[str] = Field(default_factory=list)
    facts_focus: list[str] = Field(default_factory=list)
    news_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


class EvidenceSelection(BaseModel):
    selected_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class CriticAssessment(BaseModel):
    overall_score: float = 0.0
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    decision: Literal["APPROVE", "REVISE_ONCE"] = "REVISE_ONCE"
    required_improvements: list[str] = Field(default_factory=list)
    relevance: float = 0.0
    coverage: float = 0.0
    source_diversity: float = 0.0
    citation_readiness: float = 0.0
    contradiction_risk: float = 0.8
    summary_text: str = ""


class ComparisonDecision(BaseModel):
    strategy: Literal["PICK_PASS_1", "PICK_PASS_2", "HYBRID"] = "PICK_PASS_1"
    rationale: str = ""


@dataclass
class AgentRunResult:
    summary: str
    memo_markdown: str
    confidence: str
    confidence_score: float
    evidence: list[dict[str, Any]]
    critic_notes: list[str]
    trace: list[dict[str, Any]]
    writer_model: str
    token_usage: dict[str, int]
    estimated_cost_usd: float


class SwarmState(TypedDict, total=False):
    query: str
    ticker: Optional[str]
    top_k: int
    max_evidence: int
    include_trace: bool
    trace: list[dict[str, Any]]

    pass1: dict[str, Any]
    critique1: dict[str, Any]
    pass2: dict[str, Any]
    critique2: dict[str, Any]

    needs_revision: bool
    revision_used: bool

    comparison: dict[str, Any]
    final: dict[str, Any]


class LangGraphSwarmWorkflow:
    SOURCE_WEIGHTS = {
        "filings": 1.0,
        "sec_facts": 0.9,
        "news": 0.6,
    }

    TICKER_TO_CIK = {
        "NVDA": "0001045810",
        "MSFT": "0000789019",
        "AAPL": "0000320193",
        "AMZN": "0001018724",
        "GOOGL": "0001652044",
        "META": "0001326801",
        "TSLA": "0001318605",
        "JPM": "0000019617",
        "GS": "0000886982",
        "BRK-B": "0001067983",
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger("ai_research.agents.langgraph")
        self.indexer = indexer
        if StateGraph is None or ChatOpenAI is None:
            raise RuntimeError("langgraph/langchain-openai must be installed for agent workflow.")
        self.llm = self._build_llm()
        if self.llm is None:
            raise RuntimeError("OPENAI_API_KEY is required for LangGraph agent workflow.")
        self.news_client = TavilyNewsClient()
        self.graph = self._build_graph()

    def analyze(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: Optional[int] = None,
        max_evidence: Optional[int] = None,
        include_trace: bool = True,
    ) -> AgentRunResult:
        state: SwarmState = {
            "query": query,
            "ticker": ticker,
            "top_k": top_k or settings.agent_top_k_default,
            "max_evidence": max_evidence or settings.agent_max_evidence_default,
            "include_trace": include_trace,
            "trace": [],
            "revision_used": False,
        }

        out = self.graph.invoke(state)
        final = out.get("final", {})
        selected_evidence = final.get("selected_evidence", [])
        confidence = str(final.get("confidence", "LOW"))
        confidence_score = self._confidence_score(confidence)

        return AgentRunResult(
            summary=str(final.get("summary", "Agentic memo generated")),
            memo_markdown=str(final.get("memo_markdown", "")),
            confidence=confidence,
            confidence_score=confidence_score,
            evidence=selected_evidence,
            critic_notes=final.get("critic_notes", []),
            trace=out.get("trace", []) if include_trace else [],
            writer_model=str(final.get("writer_model", settings.openai_chat_model)),
            token_usage=final.get("token_usage", {}),
            estimated_cost_usd=float(final.get("estimated_cost_usd", 0.0)),
        )

    def _build_llm(self):
        if not settings.openai_api_key or ChatOpenAI is None:
            return None
        return ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout_sec,
        )

    def _build_graph(self):
        builder = StateGraph(SwarmState)

        builder.add_node("research_pass1", self._node_research_pass1)
        builder.add_node("critic_eval1", self._node_critic_eval1)
        builder.add_node("research_pass2", self._node_research_pass2)
        builder.add_node("critic_eval2", self._node_critic_eval2)
        builder.add_node("comparison", self._node_comparison)
        builder.add_node("writer", self._node_writer)

        builder.add_edge(START, "research_pass1")
        builder.add_edge("research_pass1", "critic_eval1")
        builder.add_conditional_edges(
            "critic_eval1",
            self._route_after_critic1,
            {
                "research_pass2": "research_pass2",
                "comparison": "comparison",
            },
        )
        builder.add_edge("research_pass2", "critic_eval2")
        builder.add_edge("critic_eval2", "comparison")
        builder.add_edge("comparison", "writer")
        builder.add_edge("writer", END)

        return builder.compile()

    def _route_after_critic1(self, state: SwarmState) -> Literal["research_pass2", "comparison"]:
        if state.get("needs_revision") and not state.get("revision_used", False):
            return "research_pass2"
        return "comparison"

    def _node_research_pass1(self, state: SwarmState) -> SwarmState:
        self.logger.info("agent_start agent=research_swarm pass=1 ticker=%s query=%s", state.get("ticker"), state["query"][:160])
        result = self._run_swarm_research(
            query=state["query"],
            ticker=state.get("ticker"),
            top_k=state["top_k"],
            max_evidence=state["max_evidence"],
            feedback=None,
        )
        trace = state.get("trace", [])
        trace.extend(self._trace_research_subagents(result, "pass1"))
        trace.append({"agent": "research_swarm_pass1", "output": self._trace_research(result)})
        self.logger.info(
            "agent_done agent=research_swarm pass=1 evidence_count=%s source_counts=%s",
            len(result.get("evidence", [])),
            result.get("source_counts", {}),
        )
        return {"pass1": result, "trace": trace}

    def _node_critic_eval1(self, state: SwarmState) -> SwarmState:
        self.logger.info("agent_start agent=critic pass=1")
        critique = self._critic_evaluate(
            query=state["query"],
            candidate=state.get("pass1", {}),
            previous=None,
        )
        needs_revision = critique.get("decision") == "REVISE_ONCE" and settings.agent_swarm_revision_max >= 1
        trace = state.get("trace", [])
        trace.append({"agent": "critic_eval_pass1", "output": critique})
        self.logger.info(
            "agent_done agent=critic pass=1 decision=%s score=%s confidence=%s",
            critique.get("decision"),
            critique.get("overall_score"),
            critique.get("confidence"),
        )
        return {
            "critique1": critique,
            "needs_revision": needs_revision,
            "trace": trace,
        }

    def _node_research_pass2(self, state: SwarmState) -> SwarmState:
        self.logger.info("agent_start agent=research_swarm pass=2")
        feedback = "; ".join(state.get("critique1", {}).get("required_improvements", []))
        result = self._run_swarm_research(
            query=state["query"],
            ticker=state.get("ticker"),
            top_k=state["top_k"],
            max_evidence=state["max_evidence"],
            feedback=feedback,
        )
        trace = state.get("trace", [])
        trace.extend(self._trace_research_subagents(result, "pass2"))
        trace.append({"agent": "research_swarm_pass2", "output": self._trace_research(result)})
        self.logger.info(
            "agent_done agent=research_swarm pass=2 evidence_count=%s source_counts=%s",
            len(result.get("evidence", [])),
            result.get("source_counts", {}),
        )
        return {
            "pass2": result,
            "revision_used": True,
            "trace": trace,
        }

    def _node_critic_eval2(self, state: SwarmState) -> SwarmState:
        self.logger.info("agent_start agent=critic pass=2")
        critique = self._critic_evaluate(
            query=state["query"],
            candidate=state.get("pass2", {}),
            previous=state.get("critique1"),
        )
        trace = state.get("trace", [])
        trace.append({"agent": "critic_eval_pass2", "output": critique})
        self.logger.info(
            "agent_done agent=critic pass=2 decision=%s score=%s confidence=%s",
            critique.get("decision"),
            critique.get("overall_score"),
            critique.get("confidence"),
        )
        return {
            "critique2": critique,
            "trace": trace,
        }

    def _node_comparison(self, state: SwarmState) -> SwarmState:
        self.logger.info("agent_start agent=comparison")
        comparison = self._compare_candidates(
            pass1=state.get("pass1", {}),
            critique1=state.get("critique1", {}),
            pass2=state.get("pass2"),
            critique2=state.get("critique2"),
        )
        trace = state.get("trace", [])
        trace.append({"agent": "comparison_agent", "output": comparison})
        self.logger.info(
            "agent_done agent=comparison strategy=%s chosen_score=%s",
            comparison.get("strategy"),
            comparison.get("chosen_score"),
        )
        return {"comparison": comparison, "trace": trace}

    def _node_writer(self, state: SwarmState) -> SwarmState:
        self.logger.info("agent_start agent=writer")
        final = self._write_final(
            query=state["query"],
            ticker=state.get("ticker"),
            pass1=state.get("pass1", {}),
            pass2=state.get("pass2"),
            critique1=state.get("critique1", {}),
            critique2=state.get("critique2"),
            comparison=state.get("comparison", {}),
        )
        trace = state.get("trace", [])
        trace.append(
            {
                "agent": "writer",
                "output": {
                    "summary": final.get("summary"),
                    "summary_text": final.get("summary"),
                    "strategy": final.get("strategy"),
                    "writer_model": final.get("writer_model"),
                    "token_usage": final.get("token_usage", {}),
                    "agent_token_usage": final.get("agent_token_usage", {}),
                    "agent_estimated_cost_usd": final.get("agent_estimated_cost_usd", 0.0),
                    "estimated_cost_usd": final.get("estimated_cost_usd", 0.0),
                },
            }
        )
        self.logger.info(
            "agent_done agent=writer model=%s total_tokens=%s estimated_cost_usd=%s",
            final.get("writer_model"),
            (final.get("token_usage", {}) or {}).get("overall_total_tokens", (final.get("token_usage", {}) or {}).get("total_tokens")),
            final.get("estimated_cost_usd"),
        )
        return {"final": final, "trace": trace}

    def _run_swarm_research(
        self,
        query: str,
        ticker: Optional[str],
        top_k: int,
        max_evidence: int,
        feedback: Optional[str],
    ) -> dict[str, Any]:
        base_query = query if not feedback else f"{query}. Critic feedback to address: {feedback}"
        plan = self._plan_research(base_query, ticker)
        self.logger.info(
            "agent_plan ticker=%s filing_queries=%s facts_focus=%s news_queries=%s",
            ticker,
            plan.filing_queries,
            plan.facts_focus,
            plan.news_queries,
        )

        filing_queries = plan.filing_queries or [base_query]
        news_queries = plan.news_queries or [f"{ticker or ''} {base_query}".strip()]

        # Independent specialist agents run in parallel; aggregator waits for all.
        self.logger.info(
            "agent_parallel_start ticker=%s query=%s specialists=%s",
            ticker,
            query[:160],
            ["filing", "sec_facts", "news" if settings.agent_swarm_enable_news else "news_disabled"],
        )
        with ThreadPoolExecutor(max_workers=3) as pool:
            filing_future = pool.submit(
                self._run_specialist_with_logging,
                "filing_agent",
                self._filing_agent,
                base_query,
                ticker,
                top_k,
                filing_queries,
            )
            facts_future = pool.submit(
                self._run_specialist_with_logging,
                "sec_facts_agent",
                self._sec_facts_agent,
                base_query,
                ticker,
                plan.facts_focus,
            )
            if settings.agent_swarm_enable_news:
                news_future = pool.submit(
                    self._run_specialist_with_logging,
                    "news_agent",
                    self._news_agent,
                    base_query,
                    ticker,
                    news_queries,
                )
            else:
                news_future = None

            filing_evidence_raw = filing_future.result()
            facts_evidence_raw = facts_future.result()
            if news_future is not None:
                news_result = news_future.result()
            else:
                news_result = NewsFetchResult(
                    provider=settings.news_provider,
                    rows=[],
                    error="news_disabled",
                )
            news_evidence_raw = news_result.rows
        self.logger.info(
            "agent_parallel_done ticker=%s filings=%s sec_facts=%s news=%s",
            ticker,
            len(filing_evidence_raw),
            len(facts_evidence_raw),
            len(news_evidence_raw),
        )
        self.logger.info(
            "agent_raw_evidence_counts ticker=%s filings=%s sec_facts=%s news=%s news_error=%s",
            ticker,
            len(filing_evidence_raw),
            len(facts_evidence_raw),
            len(news_evidence_raw),
            news_result.error,
        )

        filing_evidence = self._select_evidence_with_agent(
            agent_name="Filing Agent",
            query=base_query,
            rows=filing_evidence_raw,
            limit=max(3, min(top_k, max_evidence)),
        )
        facts_evidence = self._select_evidence_with_agent(
            agent_name="SEC Facts Agent",
            query=base_query,
            rows=facts_evidence_raw,
            limit=min(settings.agent_swarm_sec_facts_max_items, max_evidence),
        )
        news_evidence = self._select_evidence_with_agent(
            agent_name="News Agent",
            query=base_query,
            rows=news_evidence_raw,
            limit=min(settings.agent_swarm_news_max_items, max_evidence),
        )

        aggregated = self._aggregate_evidence(
            filings=filing_evidence,
            sec_facts=facts_evidence,
            news=news_evidence,
            max_evidence=max_evidence,
        )

        filing_summary = self._summarize_agent_output(
            agent_name="Filing Agent",
            query=query,
            ticker=ticker,
            rows=filing_evidence,
            extra_context="Focus on disclosures, management commentary, and filing-grounded risk/growth factors.",
        )
        facts_summary = self._summarize_agent_output(
            agent_name="SEC Facts Agent",
            query=query,
            ticker=ticker,
            rows=facts_evidence,
            extra_context="Focus on quantitative trend signals from structured company facts.",
        )
        news_summary = self._summarize_agent_output(
            agent_name="News Agent",
            query=query,
            ticker=ticker,
            rows=news_evidence,
            extra_context="Focus on recent external developments, sentiment, and potential catalysts.",
            no_data_message="No reliable recent news retrieved.",
        )

        source_counts = {
            "filings": len(filing_evidence),
            "sec_facts": len(facts_evidence),
            "news": len(news_evidence),
        }
        coverage_notes = self._coverage_notes(source_counts)
        aggregator_summary = self._summarize_agent_output(
            agent_name="Research Aggregator",
            query=query,
            ticker=ticker,
            rows=aggregated,
            extra_context=(
                "Blend source-specific findings into one synthesis. Explicitly weigh corroboration and disagreements. "
                f"Coverage notes: {'; '.join(coverage_notes) if coverage_notes else 'balanced'}."
            ),
        )
        candidate_summary = (
            aggregator_summary.get("summary_text", "").splitlines()[0]
            if aggregator_summary.get("summary_text")
            else ""
        )
        self.logger.info(
            "agent_metrics ticker=%s filings_tokens=%s filings_cost=%s sec_facts_tokens=%s sec_facts_cost=%s news_tokens=%s news_cost=%s aggregator_tokens=%s aggregator_cost=%s",
            ticker,
            (filing_summary.get("token_usage", {}) or {}).get("total_tokens", 0),
            filing_summary.get("estimated_cost_usd", 0.0),
            (facts_summary.get("token_usage", {}) or {}).get("total_tokens", 0),
            facts_summary.get("estimated_cost_usd", 0.0),
            (news_summary.get("token_usage", {}) or {}).get("total_tokens", 0),
            news_summary.get("estimated_cost_usd", 0.0),
            (aggregator_summary.get("token_usage", {}) or {}).get("total_tokens", 0),
            aggregator_summary.get("estimated_cost_usd", 0.0),
        )
        if not candidate_summary:
            raise RuntimeError("Aggregator summary is empty; cannot continue.")
        return {
            "query": query,
            "feedback_used": feedback,
            "candidate_summary": candidate_summary,
            "source_counts": source_counts,
            "coverage_notes": coverage_notes,
            "agent_summaries": {
                "filings": filing_summary.get("summary_text", ""),
                "sec_facts": facts_summary.get("summary_text", ""),
                "news": news_summary.get("summary_text", ""),
                "aggregator": aggregator_summary.get("summary_text", ""),
            },
            "research_plan": plan.model_dump(),
            "agent_metrics": {
                "filings": filing_summary,
                "sec_facts": facts_summary,
                "news": news_summary,
                "aggregator": aggregator_summary,
            },
            "source_outputs": {
                "filings": self._evidence_preview(filing_evidence, **filing_summary),
                "sec_facts": self._evidence_preview(facts_evidence, **facts_summary),
                "news": self._evidence_preview(
                    news_evidence,
                    provider=news_result.provider,
                    error=news_result.error,
                    **news_summary,
                ),
                "aggregated": self._evidence_preview(
                    aggregated,
                    source_counts=source_counts,
                    coverage_notes=coverage_notes,
                    **aggregator_summary,
                ),
            },
            "evidence": aggregated,
        }

    def _plan_research(self, query: str, ticker: Optional[str]) -> ResearchPlan:
        planner = self.llm.with_structured_output(ResearchPlan)
        system = (
            "You are a research planning agent for equity analysis. "
            "Create concise retrieval plans across filings, SEC facts, and news."
        )
        user = textwrap.dedent(
            f"""
            Query: {query}
            Ticker: {ticker or 'UNKNOWN'}
            Return targeted short queries (2-3 each source) and facts focus terms.
            """
        ).strip()
        out = planner.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if not isinstance(out, ResearchPlan):
            raise RuntimeError("Research planner returned invalid output.")
        out.filing_queries = [q for q in out.filing_queries if q][:3]
        out.news_queries = [q for q in out.news_queries if q][:3]
        if not out.filing_queries or not out.news_queries or not out.facts_focus:
            raise RuntimeError("Research planner returned incomplete plan.")
        return out

    def _filing_agent(self, query: str, ticker: Optional[str], top_k: int, variants: list[str]) -> list[dict[str, Any]]:
        variants = variants[:4] if variants else [query]

        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for q in variants:
            for row in self.indexer.search(q, top_k=top_k, ticker=ticker):
                cid = row.get("chunk_id", "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                row["source_type"] = "filings"
                rows.append(row)

        rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        self.logger.info("agent_run agent=filing_retriever ticker=%s query_variants=%s selected=%s", ticker, len(variants), min(len(rows), max(1, top_k)))
        return rows[: max(1, top_k)]

    def _sec_facts_agent(self, query: str, ticker: Optional[str], focus_terms: list[str]) -> list[dict[str, Any]]:
        if not ticker:
            return []
        cik = self.TICKER_TO_CIK.get(ticker.upper())
        if not cik:
            return []

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        headers = {"User-Agent": settings.sec_user_agent, "Accept": "application/json"}
        try:
            with httpx.Client(timeout=settings.news_timeout_sec, headers=headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return []

        facts = data.get("facts", {}).get("us-gaap", {})
        tags = [
            "Revenues",
            "GrossProfit",
            "OperatingIncomeLoss",
            "NetIncomeLoss",
            "EarningsPerShareBasic",
            "CashAndCashEquivalentsAtCarryingValue",
        ]

        rows: list[dict[str, Any]] = []
        idx = 0
        for tag in tags:
            units = facts.get(tag, {}).get("units", {})
            if not units:
                continue
            series = next(iter(units.values()), [])
            if not series:
                continue
            latest = series[-1]
            val = latest.get("val")
            end = latest.get("end")
            if val is None:
                continue

            text = f"SEC Fact {tag}: {val} (period_end={end})"
            focus_blob = " ".join(focus_terms or [])
            score = 0.78 if re.search(re.escape(tag), focus_blob, re.IGNORECASE) else 0.72
            rows.append(
                {
                    "chunk_id": f"facts-{ticker}-{tag}-{idx}",
                    "accession_no": f"SEC_FACTS:{cik}",
                    "ticker": ticker.upper(),
                    "form_type": "SEC_FACTS",
                    "section_name": tag,
                    "chunk_index": idx,
                    "text_preview": text,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "score": score,
                    "source_type": "sec_facts",
                }
            )
            idx += 1
            if idx >= settings.agent_swarm_sec_facts_max_items:
                break

        self.logger.info("agent_run agent=sec_facts_retriever ticker=%s focus_terms=%s selected=%s", ticker, focus_terms, len(rows))
        return rows

    def _news_agent(self, query: str, ticker: Optional[str], query_variants: list[str]) -> NewsFetchResult:
        if not ticker:
            return NewsFetchResult(provider=settings.news_provider, rows=[], error="missing_ticker")

        if settings.news_provider.lower() != "tavily":
            raise RuntimeError("NEWS_PROVIDER must be 'tavily' for the current implementation.")

        merged_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for q in (query_variants or [])[:3]:
            res = self.news_client.fetch(q, ticker=ticker.upper())
            if res.error:
                errors.append(res.error)
            merged_rows.extend(res.rows)

        if merged_rows:
            dedup: dict[str, dict[str, Any]] = {}
            for row in merged_rows:
                key = str(row.get("chunk_id") or row.get("source_url") or "")
                if not key:
                    continue
                current = dedup.get(key)
                if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                    dedup[key] = row
            rows = sorted(dedup.values(), key=lambda r: float(r.get("score", 0.0)), reverse=True)
            final_rows = rows[: max(1, settings.agent_swarm_news_max_items)]
            self.logger.info("agent_run agent=news_retriever ticker=%s query_variants=%s selected=%s", ticker, len(query_variants or []), len(final_rows))
            return NewsFetchResult(provider="tavily", rows=final_rows, error=None)

        err = "; ".join(errors) if errors else "no_results"
        self.logger.info("agent_run agent=news_retriever ticker=%s query_variants=%s selected=0 error=%s", ticker, len(query_variants or []), err)
        return NewsFetchResult(provider="tavily", rows=[], error=err)

    def _run_specialist_with_logging(self, agent_name: str, fn, *args, **kwargs):
        started = time.perf_counter()
        self.logger.info("agent_start agent=%s", agent_name)
        try:
            out = fn(*args, **kwargs)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if isinstance(out, NewsFetchResult):
                count = len(out.rows)
                extra = f" provider={out.provider} error={out.error}"
            elif isinstance(out, list):
                count = len(out)
                extra = ""
            else:
                count = -1
                extra = ""
            self.logger.info(
                "agent_done agent=%s duration_ms=%s result_count=%s%s",
                agent_name,
                elapsed_ms,
                count,
                extra,
            )
            return out
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.logger.exception(
                "agent_failed agent=%s duration_ms=%s error=%s",
                agent_name,
                elapsed_ms,
                str(exc),
            )
            raise

    def _select_evidence_with_agent(
        self,
        agent_name: str,
        query: str,
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []

        top_rows = rows[: min(20, len(rows))]
        id_to_row = {str(r.get("chunk_id", "")): r for r in top_rows if r.get("chunk_id")}
        if not id_to_row:
            raise RuntimeError(f"{agent_name} candidates do not contain chunk IDs.")
        evidence_lines = []
        for i, row in enumerate(top_rows, start=1):
            evidence_lines.append(
                f"id={row.get('chunk_id')} source={row.get('source_type')} score={row.get('score')} "
                f"form={row.get('form_type')} section={row.get('section_name')} preview={str(row.get('text_preview',''))[:200]}"
            )
        selector = self.llm.with_structured_output(EvidenceSelection)
        valid_ids = list(id_to_row.keys())
        system = (
            f"You are {agent_name}. Select the best evidence IDs for the query. "
            "Choose only IDs from the provided list and prioritize grounding and relevance. "
            "Do not invent IDs."
        )
        attempts = 2
        selected: list[dict[str, Any]] = []
        for attempt in range(attempts):
            user = textwrap.dedent(
                f"""
                Query: {query}
                Max IDs to select: {max(1, limit)}
                Valid IDs: {valid_ids}
                Candidates:
                {'\n'.join(evidence_lines)}
                """
            ).strip()
            out = selector.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            if not isinstance(out, EvidenceSelection):
                continue
            seen = set()
            selected = []
            for cid in out.selected_ids:
                cid = str(cid)
                if cid in id_to_row and cid not in seen:
                    selected.append(id_to_row[cid])
                    seen.add(cid)
                if len(selected) >= max(1, limit):
                    break
            if selected:
                return selected
            # Retry once with an explicit correction nudge.
            system = (
                f"You are {agent_name}. Your prior output used invalid IDs. "
                "Return only IDs from Valid IDs list."
            )

        # Safety guardrail: avoid hard failure when model returns invalid IDs.
        return top_rows[: max(1, limit)]

    def _aggregate_evidence(self, **kwargs: Any) -> list[dict[str, Any]]:
        max_evidence = int(kwargs.pop("max_evidence"))
        rows: list[dict[str, Any]] = []
        for source, items in kwargs.items():
            weight = self.SOURCE_WEIGHTS.get(source, 0.5)
            for item in items:
                base = float(item.get("score", 0.0))
                item = dict(item)
                item["score"] = round(min(1.0, base * weight), 4)
                rows.append(item)

        rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

        dedup: list[dict[str, Any]] = []
        seen = set()
        for row in rows:
            cid = row.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            dedup.append(row)
            if len(dedup) >= max(1, min(max_evidence, 25)):
                break
        return dedup

    def _critic_evaluate(
        self,
        query: str,
        candidate: dict[str, Any],
        previous: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence = candidate.get("evidence", [])
        if not evidence:
            return {
                "overall_score": 0.0,
                "confidence": "LOW",
                "decision": "REVISE_ONCE",
                "summary_text": "Critic found insufficient evidence quality and requested one revision.",
                "required_improvements": ["No evidence retrieved; broaden and re-target research queries."],
                "subscores": {
                    "relevance": 0.0,
                    "coverage": 0.0,
                    "source_diversity": 0.0,
                    "citation_readiness": 0.0,
                    "contradiction_risk": 0.8,
                },
                "delta_vs_previous": None,
            }
        delta = None
        if previous and isinstance(previous.get("overall_score"), (int, float)):
            delta = round(float(previous["overall_score"]), 4)

        critic = self.llm.with_structured_output(CriticAssessment)
        evidence_preview = "\n".join(
            f"- source={e.get('source_type')} score={e.get('score')} form={e.get('form_type')} "
            f"section={e.get('section_name')} preview={str(e.get('text_preview',''))[:180]}"
            for e in evidence[:12]
        )
        crit_user = textwrap.dedent(
            f"""
            Query: {query}
            Candidate summary: {candidate.get('candidate_summary', '')}
            Source counts: {candidate.get('source_counts', {})}
            Prior score: {previous.get('overall_score') if previous else 'None'}
            Evidence preview:
            {evidence_preview}
            """
        ).strip()
        crit = critic.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict evaluator agent for equity-research outputs. "
                        "Score quality 0..1 and decide APPROVE or REVISE_ONCE. "
                        "Use only provided evidence; do not hallucinate."
                    )
                ),
                HumanMessage(content=crit_user),
            ]
        )
        out = crit.model_dump()
        out["overall_score"] = round(max(0.0, min(1.0, float(out.get("overall_score", 0.0)))), 4)
        out["subscores"] = {
            "relevance": round(float(out.pop("relevance", 0.0)), 4),
            "coverage": round(float(out.pop("coverage", 0.0)), 4),
            "source_diversity": round(float(out.pop("source_diversity", 0.0)), 4),
            "citation_readiness": round(float(out.pop("citation_readiness", 0.0)), 4),
            "contradiction_risk": round(float(out.pop("contradiction_risk", 0.8)), 4),
        }
        out["delta_vs_previous"] = (
            round(out["overall_score"] - float(previous["overall_score"]), 4)
            if previous and isinstance(previous.get("overall_score"), (int, float))
            else None
        )
        if not out.get("summary_text"):
            out["summary_text"] = (
                f"Critic confidence is {out.get('confidence', 'LOW')} "
                f"(score={out.get('overall_score')})."
            )
        return out

    def _compare_candidates(
        self,
        pass1: dict[str, Any],
        critique1: dict[str, Any],
        pass2: Optional[dict[str, Any]],
        critique2: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        score1 = float((critique1 or {}).get("overall_score", 0.0))
        if not pass2 or not critique2:
            return {
                "strategy": "PICK_PASS_1",
                "chosen_score": score1,
                "rationale": "Only one candidate available or revision not triggered.",
                "summary_text": "Comparison selected the first research result.",
                "selected_evidence": pass1.get("evidence", []),
            }

        score2 = float((critique2 or {}).get("overall_score", 0.0))
        ev1 = pass1.get("evidence", [])
        ev2 = pass2.get("evidence", [])
        comparator = self.llm.with_structured_output(ComparisonDecision)
        cmp_user = textwrap.dedent(
            f"""
            Pass1 score: {score1}
            Pass2 score: {score2}
            Pass1 evidence count: {len(ev1)}
            Pass2 evidence count: {len(ev2)}
            Pass1 summary: {pass1.get('candidate_summary', '')}
            Pass2 summary: {pass2.get('candidate_summary', '')}

            Choose strategy:
            - PICK_PASS_1
            - PICK_PASS_2
            - HYBRID
            """
        ).strip()
        dec = comparator.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a comparison agent selecting the best research candidate. "
                        "Pick the strategy that maximizes evidence quality and coverage."
                    )
                ),
                HumanMessage(content=cmp_user),
            ]
        )
        strategy = dec.strategy
        if strategy == "PICK_PASS_2":
            selected = ev2
            chosen = score2
            summary_text = "Comparison selected the revised result."
        elif strategy == "HYBRID":
            selected = self._aggregate_evidence(
                filings=ev1,
                sec_facts=ev2,
                news=[],
                max_evidence=settings.agent_max_evidence_default,
            )
            chosen = round(max(score1, score2), 4)
            summary_text = "Comparison merged both passes for broader coverage."
        else:
            selected = ev1
            chosen = score1
            summary_text = "Comparison selected the first research result."
        return {
            "strategy": strategy,
            "chosen_score": chosen,
            "rationale": dec.rationale or "Comparator decision.",
            "summary_text": summary_text,
            "selected_evidence": selected,
        }

    def _write_final(
        self,
        query: str,
        ticker: Optional[str],
        pass1: dict[str, Any],
        pass2: Optional[dict[str, Any]],
        critique1: dict[str, Any],
        critique2: Optional[dict[str, Any]],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        strategy = comparison.get("strategy", "PICK_PASS_1")
        selected = comparison.get("selected_evidence", pass1.get("evidence", []))
        chosen_score = float(comparison.get("chosen_score", 0.0))
        confidence = "HIGH" if chosen_score >= 0.75 else ("MEDIUM" if chosen_score >= 0.5 else "LOW")
        agent_totals = self._aggregate_agent_metrics(pass1, pass2)

        memo, usage = self._write_with_langchain(
            query=query,
            ticker=ticker,
            strategy=strategy,
            selected=selected,
            critique1=critique1,
            critique2=critique2,
            pass1_summary=str(pass1.get("candidate_summary", "")),
            pass2_summary=str((pass2 or {}).get("candidate_summary", "")),
            pass1_agent_summaries=pass1.get("agent_summaries", {}),
            pass2_agent_summaries=(pass2 or {}).get("agent_summaries", {}),
        )
        token_usage = {
            "prompt_tokens": int((usage or {}).get("input_tokens", 0) or 0),
            "completion_tokens": int((usage or {}).get("output_tokens", 0) or 0),
            "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
        }
        if token_usage["total_tokens"] <= 0:
            token_usage = self._estimate_tokens(query, selected, memo)
        cost = self._estimate_cost_usd(token_usage)
        total_cost = round(cost + float(agent_totals.get("estimated_cost_usd", 0.0)), 6)
        summary = memo.splitlines()[0].strip("# ") if memo else "Agentic memo generated"
        critic_notes = [
            f"Pass1 score={critique1.get('overall_score', 0.0)}",
            f"Pass2 score={(critique2 or {}).get('overall_score', 0.0)}" if critique2 else "Pass2 not executed",
            str(comparison.get("rationale", "")),
        ]
        combined_usage = dict(token_usage)
        combined_usage["agent_summary_prompt_tokens"] = int(agent_totals.get("prompt_tokens", 0))
        combined_usage["agent_summary_completion_tokens"] = int(agent_totals.get("completion_tokens", 0))
        combined_usage["agent_summary_total_tokens"] = int(agent_totals.get("total_tokens", 0))
        combined_usage["overall_total_tokens"] = int(
            int(combined_usage.get("total_tokens", 0))
            + int(combined_usage.get("agent_summary_total_tokens", 0))
        )
        return {
            "summary": summary[:200],
            "memo_markdown": memo,
            "strategy": strategy,
            "selected_evidence": selected,
            "confidence": confidence,
            "critic_notes": critic_notes,
            "writer_model": settings.openai_chat_model,
            "token_usage": combined_usage,
            "agent_token_usage": agent_totals.get("by_agent", {}),
            "agent_estimated_cost_usd": float(agent_totals.get("estimated_cost_usd", 0.0)),
            "estimated_cost_usd": total_cost,
        }

    def _write_with_langchain(
        self,
        query: str,
        ticker: Optional[str],
        strategy: str,
        selected: list[dict[str, Any]],
        critique1: dict[str, Any],
        critique2: Optional[dict[str, Any]],
        pass1_summary: str,
        pass2_summary: str,
        pass1_agent_summaries: dict[str, Any],
        pass2_agent_summaries: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        evidence_lines = []
        for i, e in enumerate(selected, start=1):
            evidence_lines.append(
                f"[{i}] source={e.get('source_type','unknown')} ticker={e.get('ticker')} form={e.get('form_type')} "
                f"section={e.get('section_name')} score={e.get('score')} preview={e.get('text_preview')}"
            )

        system = (
            "You are a buy-side equity research assistant. Produce concise, evidence-grounded output. "
            "Every substantive bullet in Executive Summary, Bull Case Signals, and Risk Signals must include citations like [1], [2]."
        )
        user = textwrap.dedent(
            f"""
            Query: {query}
            Ticker: {ticker or 'NONE'}
            Strategy selected by comparison agent: {strategy}
            Pass1 summary: {pass1_summary}
            Pass2 summary: {pass2_summary}
            Pass1 filing summary: {pass1_agent_summaries.get('filings', '')}
            Pass1 facts summary: {pass1_agent_summaries.get('sec_facts', '')}
            Pass1 news summary: {pass1_agent_summaries.get('news', '')}
            Pass1 aggregator summary: {pass1_agent_summaries.get('aggregator', '')}
            Pass2 filing summary: {pass2_agent_summaries.get('filings', '')}
            Pass2 facts summary: {pass2_agent_summaries.get('sec_facts', '')}
            Pass2 news summary: {pass2_agent_summaries.get('news', '')}
            Pass2 aggregator summary: {pass2_agent_summaries.get('aggregator', '')}
            Pass1 score: {critique1.get('overall_score', 0.0)}
            Pass2 score: {(critique2 or {}).get('overall_score', 0.0)}

            Evidence:
            {'\n'.join(evidence_lines)}

            Return markdown with sections:
            1) Executive Summary (3 bullets)
            2) Bull Case Signals
            3) Risk Signals
            4) What To Verify Next (3 bullets)
            5) Citations
            """
        ).strip()

        msg = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = str(msg.content or "").strip()
        usage = getattr(msg, "usage_metadata", None) or {}
        return content, usage

    def _estimate_tokens(self, query: str, evidence: list[dict[str, Any]], memo: str) -> dict[str, int]:
        chars_per_token = max(1, settings.token_chars_per_token)
        evidence_chars = sum(len(str(e.get("text_preview", ""))) for e in evidence)
        prompt_tokens = max(1, (len(query) + evidence_chars) // chars_per_token)
        completion_tokens = max(1, len(memo) // chars_per_token) if memo else 0
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        }

    def _estimate_cost_usd(self, token_usage: dict[str, int]) -> float:
        prompt_tokens = float(token_usage.get("prompt_tokens", 0))
        completion_tokens = float(token_usage.get("completion_tokens", 0))
        in_cost = (prompt_tokens / 1000.0) * settings.openai_chat_input_cost_per_1k
        out_cost = (completion_tokens / 1000.0) * settings.openai_chat_output_cost_per_1k
        return round(in_cost + out_cost, 6)

    def _confidence_score(self, confidence: str) -> float:
        return {
            "LOW": 0.33,
            "MEDIUM": 0.66,
            "HIGH": 1.0,
        }.get(str(confidence).upper(), 0.0)

    def _aggregate_agent_metrics(self, pass1: dict[str, Any], pass2: Optional[dict[str, Any]]) -> dict[str, Any]:
        by_agent: dict[str, dict[str, Any]] = {}
        for label, source in (("pass1", pass1), ("pass2", pass2 or {})):
            metrics = (source or {}).get("agent_metrics", {}) if isinstance(source, dict) else {}
            if not isinstance(metrics, dict):
                continue
            for agent_name, details in metrics.items():
                if not isinstance(details, dict):
                    continue
                key = f"{agent_name}_{label}"
                usage = details.get("token_usage", {}) or {}
                by_agent[key] = {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    "estimated_cost_usd": float(details.get("estimated_cost_usd", 0.0) or 0.0),
                }
        return {
            "by_agent": by_agent,
            "prompt_tokens": sum(int(v.get("prompt_tokens", 0)) for v in by_agent.values()),
            "completion_tokens": sum(int(v.get("completion_tokens", 0)) for v in by_agent.values()),
            "total_tokens": sum(int(v.get("total_tokens", 0)) for v in by_agent.values()),
            "estimated_cost_usd": round(sum(float(v.get("estimated_cost_usd", 0.0)) for v in by_agent.values()), 6),
        }

    def _coverage_notes(self, source_counts: dict[str, int]) -> list[str]:
        notes: list[str] = []
        if source_counts.get("filings", 0) <= 0:
            notes.append("No filing evidence available.")
        if source_counts.get("sec_facts", 0) <= 0:
            notes.append("No structured SEC facts available.")
        if source_counts.get("news", 0) <= 0:
            notes.append("No reliable recent news retrieved.")
        if not notes:
            notes.append("Coverage is balanced across available sources.")
        return notes

    def _summarize_agent_output(
        self,
        agent_name: str,
        query: str,
        ticker: Optional[str],
        rows: list[dict[str, Any]],
        extra_context: str,
        no_data_message: str = "No strong evidence found for this source.",
    ) -> dict[str, Any]:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not rows:
            self.logger.info(
                "agent_summary agent=%s ticker=%s evidence_count=0 total_tokens=0 estimated_cost_usd=0.0",
                agent_name,
                ticker,
            )
            return {
                "summary_text": no_data_message,
                "token_usage": token_usage,
                "estimated_cost_usd": 0.0,
            }

        top_rows = rows[: min(5, len(rows))]
        evidence_lines = []
        for idx, row in enumerate(top_rows, start=1):
            evidence_lines.append(
                f"[{idx}] source={row.get('source_type')} form={row.get('form_type')} "
                f"section={row.get('section_name')} score={row.get('score')} "
                f"preview={str(row.get('text_preview', ''))[:180]}"
            )

        system = (
            "You are a financial research assistant. Return 3 concise bullets. "
            "Only use supplied evidence. Include citation tags like [1], [2] in each bullet."
        )
        user = textwrap.dedent(
            f"""
            Agent: {agent_name}
            Query: {query}
            Ticker: {ticker or 'NONE'}
            Context: {extra_context}

            Evidence:
            {'\n'.join(evidence_lines)}
            """
        ).strip()
        msg = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = str(msg.content or "").strip()
        usage = getattr(msg, "usage_metadata", None) or {}
        token_usage = {
            "prompt_tokens": int((usage or {}).get("input_tokens", 0) or 0),
            "completion_tokens": int((usage or {}).get("output_tokens", 0) or 0),
            "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
        }
        if token_usage["total_tokens"] <= 0:
            token_usage = self._estimate_tokens(query, top_rows, content)
        if not content:
            raise RuntimeError(f"{agent_name} returned empty summary")
        estimated_cost = self._estimate_cost_usd(token_usage)
        self.logger.info(
            "agent_summary agent=%s ticker=%s evidence_count=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s estimated_cost_usd=%s",
            agent_name,
            ticker,
            len(rows),
            token_usage.get("prompt_tokens", 0),
            token_usage.get("completion_tokens", 0),
            token_usage.get("total_tokens", 0),
            estimated_cost,
        )
        return {
            "summary_text": content,
            "token_usage": token_usage,
            "estimated_cost_usd": estimated_cost,
        }

    def _evidence_preview(self, rows: list[dict[str, Any]], limit: int = 3, **extras: Any) -> dict[str, Any]:
        preview = []
        for row in rows[: max(1, limit)]:
            preview.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "ticker": row.get("ticker"),
                    "form_type": row.get("form_type"),
                    "section_name": row.get("section_name"),
                    "score": row.get("score"),
                    "text_preview": str(row.get("text_preview", ""))[:160],
                    "source_type": row.get("source_type"),
                    "source_url": row.get("source_url"),
                }
            )
        out = {"count": len(rows), "top_items": preview}
        out.update(extras)
        return out

    def _trace_research_subagents(self, result: dict[str, Any], pass_label: str) -> list[dict[str, Any]]:
        sources = result.get("source_outputs", {})
        return [
            {
                "agent": f"filing_agent_{pass_label}",
                "output": sources.get("filings", {"count": 0, "top_items": []}),
            },
            {
                "agent": f"sec_facts_agent_{pass_label}",
                "output": sources.get("sec_facts", {"count": 0, "top_items": []}),
            },
            {
                "agent": f"news_agent_{pass_label}",
                "output": sources.get("news", {"count": 0, "top_items": []}),
            },
            {
                "agent": f"research_aggregator_{pass_label}",
                "output": sources.get("aggregated", {"count": 0, "top_items": []}),
            },
        ]

    def _trace_research(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_summary": result.get("candidate_summary"),
            "source_counts": result.get("source_counts", {}),
            "research_plan": result.get("research_plan", {}),
            "summary_text": result.get("agent_summaries", {}).get("aggregator", result.get("candidate_summary", "")),
            "coverage_notes": result.get("coverage_notes", []),
            "evidence_count": len(result.get("evidence", [])),
        }


workflow = LangGraphSwarmWorkflow()
