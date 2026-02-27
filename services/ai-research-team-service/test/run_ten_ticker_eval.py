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

DEFAULT_TICKERS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "GS",
    "BRK-B",
]


def _slugify_ticker(ticker: str) -> str:
    return ticker.lower().replace("-", "_")


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    log_dir = repo / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ten-ticker-eval.log"
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("ai_research.test.ten_ticker_eval")

    parser = argparse.ArgumentParser(description="Run one prompt across 10 tickers and generate eval artifacts")
    parser.add_argument(
        "--query",
        default="Summarize the top growth drivers and top risks from the latest filing.",
        help="Single prompt applied to all tickers",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers (default: the standard 10-ticker set)",
    )
    parser.add_argument("--dataset-name", default="ten_ticker_single_prompt", help="Output file prefix")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-evidence", type=int, default=12)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    eval_dir = repo / "data" / "eval-reports"
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = eval_dir / f"{args.dataset_name}_dataset.json"
    csv_path = eval_dir / f"{args.dataset_name}_eval.csv"
    summary_path = eval_dir / f"{args.dataset_name}_summary.json"

    dataset = []
    for idx, ticker in enumerate(tickers, start=1):
        dataset.append(
            {
                "id": f"{_slugify_ticker(ticker).upper()}-Q{idx:02d}",
                "ticker": ticker,
                "query": args.query,
            }
        )
    dataset_path.write_text(json.dumps(dataset, indent=2))

    logger.info(
        "ten_ticker_eval_start tickers=%s dataset=%s top_k=%s max_evidence=%s",
        tickers,
        dataset_path,
        args.top_k,
        args.max_evidence,
    )

    result = eval_runner.run(
        dataset_path=str(dataset_path),
        top_k=args.top_k,
        max_evidence=args.max_evidence,
        output_csv_path=str(csv_path),
        include_trace=True,
    )
    summary_path.write_text(json.dumps(result, indent=2))

    logger.info(
        "ten_ticker_eval_complete sample_count=%s csv=%s summary=%s",
        result.get("sampleCount"),
        csv_path,
        summary_path,
    )

    print(
        json.dumps(
            {
                "tickers": tickers,
                "sampleCount": result.get("sampleCount"),
                "metrics": result.get("metrics"),
                "qualityGate": result.get("qualityGate"),
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
