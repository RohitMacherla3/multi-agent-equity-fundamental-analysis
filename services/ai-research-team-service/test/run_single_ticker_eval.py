from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.core.config import settings
from app.eval.runner import eval_runner


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    log_dir = repo / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "single-ticker-eval.log"
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("ai_research.test.single_eval")

    parser = argparse.ArgumentParser(description="Run 1-prompt single-ticker eval and generate SVGs")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AMZN")
    parser.add_argument(
        "--query",
        default="Summarize the top growth drivers and top risks from the latest filing.",
        help="Prompt/query to evaluate",
    )
    parser.add_argument("--sample-id", default=None, help="Optional sample id")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-evidence", type=int, default=12)
    args = parser.parse_args()

    eval_dir = repo / "data" / "eval-reports"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ticker = args.ticker.upper().strip()
    sample_id = args.sample_id or f"{ticker}-Q01"
    slug = ticker.lower()

    dataset_path = eval_dir / f"single_ticker_{slug}_smoke_dataset.json"
    csv_path = eval_dir / f"single_ticker_{slug}_smoke_eval.csv"
    summary_path = eval_dir / f"single_ticker_{slug}_smoke_summary.json"

    dataset = [
        {
            "id": sample_id,
            "ticker": ticker,
            "query": args.query,
        }
    ]
    dataset_path.write_text(json.dumps(dataset, indent=2))

    result = eval_runner.run(
        dataset_path=str(dataset_path),
        top_k=args.top_k,
        max_evidence=args.max_evidence,
        output_csv_path=str(csv_path),
        include_trace=True,
    )
    logger.info(
        "single_ticker_eval_complete ticker=%s sample_count=%s csv=%s summary=%s",
        ticker,
        result.get("sampleCount"),
        csv_path,
        summary_path,
    )

    summary_path.write_text(json.dumps(result, indent=2))

    print(
        json.dumps(
            {
                "ticker": ticker,
                "sampleCount": result.get("sampleCount"),
                "metrics": result.get("metrics"),
                "qualityGate": result.get("qualityGate"),
                "errorMessage": (result.get("samples") or [{}])[0].get("errorMessage"),
                "dataset": str(dataset_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
                "svgs": result.get("svgPaths"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
