from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from app.agents.engine import get_workflow
from app.core.config import settings
from app.eval.svg_reports import generate_eval_svgs
from app.risk.guardrails import sanitize_prompt


@dataclass
class EvalSampleResult:
    sample_id: str
    ticker: str | None
    query: str
    sanitized_query: str
    blocked: bool
    blocked_reason: str | None
    pii_types: list[str]
    latency_ms: int
    evidence_count: int
    citation_coverage: float
    faithfulness_proxy: float
    retrieval_hit: int
    confidence: str
    confidence_score: float
    writer_model: str
    summary: str
    memo_markdown: str
    trace_json: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    agent_summary_prompt_tokens: int
    agent_summary_completion_tokens: int
    agent_summary_total_tokens: int
    overall_total_tokens: int
    agent_estimated_cost_usd: float
    estimated_cost_usd: float
    agent_token_usage_json: str
    news_result_count: int
    error_message: str | None


class EvaluationRunner:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ai_research.eval.runner")

    def run(
        self,
        dataset_path: str,
        top_k: int = 8,
        max_evidence: int = 12,
        quality_gate: dict[str, float] | None = None,
        output_csv_path: str | None = None,
        include_trace: bool = True,
    ) -> dict[str, Any]:
        self.logger.info(
            "eval_run_start dataset=%s top_k=%s max_evidence=%s include_trace=%s eval_assets_dir=%s",
            dataset_path,
            top_k,
            max_evidence,
            include_trace,
            settings.eval_assets_dir,
        )
        quality_gate = quality_gate or {
            "min_retrieval_hit_rate": 0.6,
            "min_citation_coverage": 0.6,
            "max_avg_latency_ms": 60000.0,
        }

        samples = self._load_samples(dataset_path)
        results: list[EvalSampleResult] = []
        written_csv_path = self._write_csv(results, output_csv_path)
        checkpoint_path = Path(written_csv_path).with_suffix(".progress.jsonl")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            sample_id = str(sample.get("id", "unknown"))
            query = str(sample.get("query", ""))
            ticker = sample.get("ticker")

            guard = sanitize_prompt(query)
            if not guard.allowed:
                results.append(
                    EvalSampleResult(
                        sample_id=sample_id,
                        ticker=ticker,
                        query=query,
                        sanitized_query="",
                        blocked=True,
                        blocked_reason=guard.blocked_reason,
                        pii_types=guard.pii_types,
                        latency_ms=0,
                        evidence_count=0,
                        citation_coverage=0.0,
                        faithfulness_proxy=0.0,
                        retrieval_hit=0,
                        confidence="LOW",
                        confidence_score=0.33,
                        writer_model="blocked",
                        summary="",
                        memo_markdown="",
                        trace_json="[]",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        agent_summary_prompt_tokens=0,
                        agent_summary_completion_tokens=0,
                        agent_summary_total_tokens=0,
                        overall_total_tokens=0,
                        agent_estimated_cost_usd=0.0,
                        estimated_cost_usd=0.0,
                        agent_token_usage_json="{}",
                        news_result_count=0,
                        error_message=guard.blocked_reason or "blocked",
                    )
                )
                self._append_checkpoint(checkpoint_path, results[-1])
                self._write_csv(results, output_csv_path)
                continue

            try:
                start = time.perf_counter()
                run = get_workflow().analyze(
                    query=guard.sanitized_text,
                    ticker=ticker,
                    top_k=top_k,
                    max_evidence=max_evidence,
                    include_trace=include_trace,
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                evidence_count = len(run.evidence)
                citation_coverage = self._citation_coverage(run.memo_markdown, evidence_count)
                faithfulness_proxy = 1.0 if evidence_count > 0 else 0.0

                agent_token_usage = self._extract_agent_token_usage(run.trace)
                news_result_count = self._extract_news_count(run.trace)

                results.append(
                    EvalSampleResult(
                        sample_id=sample_id,
                        ticker=ticker,
                        query=query,
                        sanitized_query=guard.sanitized_text,
                        blocked=False,
                        blocked_reason=None,
                        pii_types=guard.pii_types,
                        latency_ms=elapsed_ms,
                        evidence_count=evidence_count,
                        citation_coverage=citation_coverage,
                        faithfulness_proxy=faithfulness_proxy,
                        retrieval_hit=1 if evidence_count > 0 else 0,
                        confidence=run.confidence,
                        confidence_score=float(run.confidence_score),
                        writer_model=run.writer_model,
                        summary=run.summary,
                        memo_markdown=run.memo_markdown,
                        trace_json=json.dumps(run.trace, ensure_ascii=False),
                        prompt_tokens=int(run.token_usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(run.token_usage.get("completion_tokens", 0) or 0),
                        total_tokens=int(run.token_usage.get("total_tokens", 0) or 0),
                        agent_summary_prompt_tokens=int(run.token_usage.get("agent_summary_prompt_tokens", 0) or 0),
                        agent_summary_completion_tokens=int(run.token_usage.get("agent_summary_completion_tokens", 0) or 0),
                        agent_summary_total_tokens=int(run.token_usage.get("agent_summary_total_tokens", 0) or 0),
                        overall_total_tokens=int(run.token_usage.get("overall_total_tokens", 0) or 0),
                        agent_estimated_cost_usd=float(
                            (run.trace[-1].get("output", {}) if run.trace else {}).get("agent_estimated_cost_usd", 0.0) or 0.0
                        ),
                        estimated_cost_usd=float(run.estimated_cost_usd),
                        agent_token_usage_json=json.dumps(agent_token_usage, ensure_ascii=False),
                        news_result_count=news_result_count,
                        error_message=None,
                    )
                )
            except Exception as exc:
                results.append(
                    EvalSampleResult(
                        sample_id=sample_id,
                        ticker=ticker,
                        query=query,
                        sanitized_query=guard.sanitized_text,
                        blocked=False,
                        blocked_reason=None,
                        pii_types=guard.pii_types,
                        latency_ms=0,
                        evidence_count=0,
                        citation_coverage=0.0,
                        faithfulness_proxy=0.0,
                        retrieval_hit=0,
                        confidence="LOW",
                        confidence_score=0.33,
                        writer_model="error",
                        summary="",
                        memo_markdown="",
                        trace_json="[]",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        agent_summary_prompt_tokens=0,
                        agent_summary_completion_tokens=0,
                        agent_summary_total_tokens=0,
                        overall_total_tokens=0,
                        agent_estimated_cost_usd=0.0,
                        estimated_cost_usd=0.0,
                        agent_token_usage_json="{}",
                        news_result_count=0,
                        error_message=str(exc),
                    )
                )
            self._append_checkpoint(checkpoint_path, results[-1])
            self._write_csv(results, output_csv_path)

        metrics = self._aggregate_metrics(results)
        gate = self._evaluate_gate(metrics, quality_gate)

        written_csv_path = self._write_csv(results, output_csv_path)
        svg_output_dir = str(Path(settings.eval_assets_dir).resolve())
        svg_paths = generate_eval_svgs(written_csv_path, metrics, svg_output_dir)
        self.logger.info(
            "eval_run_done samples=%s csv=%s checkpoint=%s svg_dir=%s",
            len(results),
            written_csv_path,
            str(checkpoint_path),
            svg_output_dir,
        )

        return {
            "datasetPath": dataset_path,
            "sampleCount": len(results),
            "metrics": metrics,
            "qualityGate": gate,
            "outputCsvPath": written_csv_path,
            "svgPaths": svg_paths,
            "samples": [self._to_dict(r) for r in results],
        }

    def _load_samples(self, dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON list")
        return data

    def _citation_coverage(self, memo_markdown: str, evidence_count: int) -> float:
        if evidence_count <= 0:
            return 0.0
        cites = 0
        for i in range(1, evidence_count + 1):
            if f"[{i}]" in memo_markdown:
                cites += 1
        return round(cites / evidence_count, 4)

    def _aggregate_metrics(self, results: list[EvalSampleResult]) -> dict[str, Any]:
        if not results:
            return {
                "request_count": 0,
                "success_count": 0,
                "error_count": 0,
                "blocked_count": 0,
                "retrieval_hit_count": 0,
                "news_coverage_count": 0,
                "total_evidence_count": 0,
                "total_latency_ms": 0,
                "total_latency_seconds": 0.0,
                "p95_latency_seconds": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_summary_input_tokens": 0,
                "total_summary_output_tokens": 0,
                "total_agent_estimated_cost_usd": 0.0,
                "total_estimated_cost_usd": 0.0,
                "citation_covered_evidence_count": 0,
                "citation_coverage_overall": 0.0,
                "retrieval_hit_rate": 0.0,
                "p95_latency_seconds": 0.0,
                "blocked_rate": 0.0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "news_coverage_rate": 0.0,
                "avg_latency_seconds": 0.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "avg_summary_input_tokens": 0.0,
                "avg_summary_output_tokens": 0.0,
                "avg_cost_per_query_usd": 0.0,
            }

        request_count = len(results)
        success_count = sum(1 for r in results if not r.error_message)
        error_count = sum(1 for r in results if r.error_message)
        blocked_count = sum(1 for r in results if r.blocked)
        retrieval_hit_count = sum(int(r.retrieval_hit) for r in results)
        news_coverage_count = sum(1 for r in results if r.news_result_count > 0)
        total_evidence_count = int(sum(r.evidence_count for r in results))
        total_latency_ms = int(sum(r.latency_ms for r in results))
        total_latency_seconds = float(total_latency_ms) / 1000.0
        total_main_input_tokens = int(sum(r.prompt_tokens for r in results))
        total_main_output_tokens = int(sum(r.completion_tokens for r in results))
        total_summary_input_tokens = int(sum(r.agent_summary_prompt_tokens for r in results))
        total_summary_output_tokens = int(sum(r.agent_summary_completion_tokens for r in results))
        total_input_tokens = total_main_input_tokens + total_summary_input_tokens
        total_output_tokens = total_main_output_tokens + total_summary_output_tokens
        total_agent_estimated_cost_usd = float(sum(r.agent_estimated_cost_usd for r in results))
        total_estimated_cost_usd = float(sum(r.estimated_cost_usd for r in results))
        avg_latency_seconds = total_latency_seconds / float(request_count)
        avg_input_tokens = float(total_input_tokens) / float(request_count)
        avg_output_tokens = float(total_output_tokens) / float(request_count)
        avg_summary_input_tokens = float(total_summary_input_tokens) / float(request_count)
        avg_summary_output_tokens = float(total_summary_output_tokens) / float(request_count)
        avg_cost_per_query_usd = total_estimated_cost_usd / float(request_count)

        citation_covered_evidence_count = int(
            round(sum(float(r.citation_coverage) * float(r.evidence_count) for r in results))
        )
        citation_coverage_overall = (
            float(citation_covered_evidence_count) / float(total_evidence_count)
            if total_evidence_count > 0
            else 0.0
        )
        hit_rate = mean(1.0 if r.evidence_count > 0 else 0.0 for r in results)
        blocked_rate = mean(1.0 if r.blocked else 0.0 for r in results)
        success_rate = mean(1.0 if not r.error_message else 0.0 for r in results)
        error_rate = mean(1.0 if r.error_message else 0.0 for r in results)
        news_coverage_rate = mean(1.0 if r.news_result_count > 0 else 0.0 for r in results)
        latencies_sec = sorted(float(r.latency_ms) / 1000.0 for r in results)
        p95_idx = max(0, int(round(0.95 * (len(latencies_sec) - 1))))
        p95_latency_seconds = latencies_sec[p95_idx] if latencies_sec else 0.0

        return {
            "request_count": request_count,
            "success_count": success_count,
            "error_count": error_count,
            "blocked_count": blocked_count,
            "retrieval_hit_count": retrieval_hit_count,
            "news_coverage_count": news_coverage_count,
            "total_evidence_count": total_evidence_count,
            "total_latency_ms": total_latency_ms,
            "total_latency_seconds": round(total_latency_seconds, 3),
            "p95_latency_seconds": round(p95_latency_seconds, 3),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_summary_input_tokens": total_summary_input_tokens,
            "total_summary_output_tokens": total_summary_output_tokens,
            "total_agent_estimated_cost_usd": round(total_agent_estimated_cost_usd, 6),
            "total_estimated_cost_usd": round(total_estimated_cost_usd, 6),
            "citation_covered_evidence_count": citation_covered_evidence_count,
            "citation_coverage_overall": round(citation_coverage_overall, 4),
            "retrieval_hit_rate": round(hit_rate, 4),
            "blocked_rate": round(blocked_rate, 4),
            "success_rate": round(success_rate, 4),
            "error_rate": round(error_rate, 4),
            "news_coverage_rate": round(news_coverage_rate, 4),
            "avg_latency_seconds": round(avg_latency_seconds, 3),
            "avg_input_tokens": round(avg_input_tokens, 2),
            "avg_output_tokens": round(avg_output_tokens, 2),
            "avg_summary_input_tokens": round(avg_summary_input_tokens, 2),
            "avg_summary_output_tokens": round(avg_summary_output_tokens, 2),
            "avg_cost_per_query_usd": round(avg_cost_per_query_usd, 6),
        }

    def _evaluate_gate(self, metrics: dict[str, Any], gate: dict[str, float]) -> dict[str, Any]:
        latency_ms_per_request = (
            float(metrics.get("total_latency_ms", 0)) / float(max(1, metrics.get("request_count", 1)))
        )
        checks = {
            "retrieval_hit_rate": metrics["retrieval_hit_rate"] >= gate["min_retrieval_hit_rate"],
            "citation_coverage_overall": metrics["citation_coverage_overall"] >= gate["min_citation_coverage"],
            "latency_ms_per_request": latency_ms_per_request <= gate["max_avg_latency_ms"],
        }
        passed = all(checks.values())
        failed_checks = [k for k, ok in checks.items() if not ok]
        return {
            "passed": passed,
            "thresholds": gate,
            "checks": checks,
            "failedChecks": failed_checks,
        }

    def _to_dict(self, sample: EvalSampleResult) -> dict[str, Any]:
        return {
            "sampleId": sample.sample_id,
            "ticker": sample.ticker,
            "query": sample.query,
            "sanitizedQuery": sample.sanitized_query,
            "blocked": sample.blocked,
            "blockedReason": sample.blocked_reason,
            "piiTypes": sample.pii_types,
            "latencyMs": sample.latency_ms,
            "latencySeconds": round(sample.latency_ms / 1000.0, 3),
            "evidenceCount": sample.evidence_count,
            "citationCoverage": sample.citation_coverage,
            "faithfulnessProxy": sample.faithfulness_proxy,
            "retrievalHit": sample.retrieval_hit,
            "confidence": sample.confidence,
            "confidenceScore": sample.confidence_score,
            "writerModel": sample.writer_model,
            "summary": sample.summary,
            "memoMarkdown": sample.memo_markdown,
            "traceJson": sample.trace_json,
            "promptTokens": sample.prompt_tokens,
            "completionTokens": sample.completion_tokens,
            "totalTokens": sample.total_tokens,
            "agentSummaryPromptTokens": sample.agent_summary_prompt_tokens,
            "agentSummaryCompletionTokens": sample.agent_summary_completion_tokens,
            "agentSummaryTotalTokens": sample.agent_summary_total_tokens,
            "overallTotalTokens": sample.overall_total_tokens,
            "agentEstimatedCostUsd": round(sample.agent_estimated_cost_usd, 6),
            "estimatedCostUsd": round(sample.estimated_cost_usd, 6),
            "agentTokenUsageJson": sample.agent_token_usage_json,
            "newsResultCount": sample.news_result_count,
            "errorMessage": sample.error_message,
        }

    def _write_csv(self, results: list[EvalSampleResult], output_csv_path: str | None) -> str:
        if output_csv_path:
            path = Path(output_csv_path)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = Path("data/eval-reports") / f"evaluation_run_{ts}.csv"

        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "sample_id",
            "ticker",
            "query",
            "sanitized_query",
            "blocked",
            "blocked_reason",
            "pii_types",
            "latency_ms",
            "latency_seconds",
            "evidence_count",
            "citation_coverage",
            "faithfulness_proxy",
            "retrieval_hit",
            "confidence",
            "confidence_score",
            "writer_model",
            "summary",
            "memo_markdown",
            "trace_json",
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "agent_summary_prompt_tokens",
            "agent_summary_completion_tokens",
            "agent_summary_total_tokens",
            "overall_total_tokens",
            "agent_estimated_cost_usd",
            "estimated_cost_usd",
            "agent_token_usage_json",
            "news_result_count",
            "error_message",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "sample_id": r.sample_id,
                        "ticker": r.ticker or "",
                        "query": r.query,
                        "sanitized_query": r.sanitized_query,
                        "blocked": r.blocked,
                        "blocked_reason": r.blocked_reason or "",
                        "pii_types": ",".join(r.pii_types),
                        "latency_ms": r.latency_ms,
                        "latency_seconds": round(r.latency_ms / 1000.0, 3),
                        "evidence_count": r.evidence_count,
                        "citation_coverage": r.citation_coverage,
                        "faithfulness_proxy": r.faithfulness_proxy,
                        "retrieval_hit": r.retrieval_hit,
                        "confidence": r.confidence,
                        "confidence_score": r.confidence_score,
                        "writer_model": r.writer_model,
                        "summary": r.summary,
                        "memo_markdown": r.memo_markdown,
                        "trace_json": r.trace_json,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "input_tokens": r.prompt_tokens,
                        "output_tokens": r.completion_tokens,
                        "total_tokens": r.total_tokens,
                        "agent_summary_prompt_tokens": r.agent_summary_prompt_tokens,
                        "agent_summary_completion_tokens": r.agent_summary_completion_tokens,
                        "agent_summary_total_tokens": r.agent_summary_total_tokens,
                        "overall_total_tokens": r.overall_total_tokens,
                        "agent_estimated_cost_usd": round(r.agent_estimated_cost_usd, 6),
                        "estimated_cost_usd": round(r.estimated_cost_usd, 6),
                        "agent_token_usage_json": r.agent_token_usage_json,
                        "news_result_count": r.news_result_count,
                        "error_message": r.error_message or "",
                    }
                )
        return str(path)

    def _append_checkpoint(self, checkpoint_path: Path, sample: EvalSampleResult) -> None:
        payload = self._to_dict(sample)
        payload["checkpointedAt"] = datetime.now(timezone.utc).isoformat()
        with checkpoint_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _extract_agent_token_usage(self, trace: list[dict[str, Any]]) -> dict[str, Any]:
        if not trace:
            return {}
        for step in reversed(trace):
            if str(step.get("agent", "")) == "writer":
                out = step.get("output", {}) if isinstance(step, dict) else {}
                value = out.get("agent_token_usage", {})
                return value if isinstance(value, dict) else {}
        return {}

    def _extract_news_count(self, trace: list[dict[str, Any]]) -> int:
        if not trace:
            return 0
        for step in trace:
            agent = str(step.get("agent", ""))
            if not agent.startswith("news_agent_"):
                continue
            out = step.get("output", {}) if isinstance(step, dict) else {}
            try:
                return int(out.get("count", 0) or 0)
            except Exception:
                return 0
        return 0


eval_runner = EvaluationRunner()
