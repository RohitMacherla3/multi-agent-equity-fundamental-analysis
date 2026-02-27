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
    log_file = log_dir / "ten-ticker-ten-prompt-eval.log"
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("ai_research.test.ten_ticker_ten_prompt_eval")

    parser = argparse.ArgumentParser(description="Run 10 tickers x 10 prompts eval and generate artifacts")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers (default: standard 10-ticker set)",
    )
    parser.add_argument(
        "--prompts-file",
        default=str(SERVICE_ROOT / "app" / "eval" / "datasets" / "analyst_questions_top10.json"),
        help="Path to JSON list of prompts",
    )
    parser.add_argument("--dataset-name", default="ten_ticker_ten_prompt", help="Output file prefix")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-evidence", type=int, default=12)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    prompts = json.loads(prompts_path.read_text())
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("Prompts file must be a JSON array of strings.")
    if len(prompts) == 0:
        raise ValueError("Prompts list is empty.")

    eval_dir = repo / "data" / "eval-reports"
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = eval_dir / f"{args.dataset_name}_dataset.json"
    csv_path = eval_dir / f"{args.dataset_name}_eval.csv"
    summary_path = eval_dir / f"{args.dataset_name}_summary.json"

    dataset = []
    idx = 1
    for ticker in tickers:
        for q_idx, query in enumerate(prompts, start=1):
            dataset.append(
                {
                    "id": f"{_slugify_ticker(ticker).upper()}-Q{q_idx:02d}-S{idx:03d}",
                    "ticker": ticker,
                    "query": query,
                }
            )
            idx += 1
    dataset_path.write_text(json.dumps(dataset, indent=2))

    logger.info(
        "ten_ticker_ten_prompt_eval_start tickers=%s prompts=%s samples=%s dataset=%s top_k=%s max_evidence=%s",
        tickers,
        len(prompts),
        len(dataset),
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
        "ten_ticker_ten_prompt_eval_complete sample_count=%s csv=%s summary=%s",
        result.get("sampleCount"),
        csv_path,
        summary_path,
    )

    print(
        json.dumps(
            {
                "tickers": tickers,
                "promptCount": len(prompts),
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

