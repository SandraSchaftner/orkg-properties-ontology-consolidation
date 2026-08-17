"""
Fetch the current ORKG property set from the live API.

This replaces the manually created property export the original OPO pipeline read from
disk (`orkg_properties_original_2025-12-31.json`). Running this right after the notebook
has downloaded its raw data guarantees that the consolidation covers every predicate the
analysis can encounter: properties are added to ORKG over time but virtually never
removed, so a dump taken *after* the analysis data is a superset of it.

Usage:
    python opo/fetch_properties.py                    # -> opo/input/orkg_properties_<today>.json
    python opo/fetch_properties.py --output some.json
    python opo/fetch_properties.py --limit 200        # small sample for testing
"""

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

API_URL = "https://orkg.org/api/predicates"
PROPERTY_URI = "https://orkg.org/property/{id}"
PAGE_SIZE = 1000
DEFAULT_DIR = Path(__file__).parent / "input"


def fetch_predicates(limit: Optional[int] = None, retries: int = 3) -> List[Dict[str, str]]:
    """Fetch all predicates, page by page, in the field layout of the original dump."""
    session = requests.Session()
    records: List[Dict[str, str]] = []
    page = 0
    bar = tqdm(desc="predicates", unit=" props")

    while True:
        for attempt in range(retries):
            try:
                response = session.get(
                    API_URL,
                    params={"page": page, "size": PAGE_SIZE},
                    headers={"Accept": "application/json"},
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)

        for predicate in payload.get("content", []):
            records.append(
                {
                    "uri": PROPERTY_URI.format(id=predicate["id"]),
                    "id": predicate["id"],
                    "label": predicate.get("label") or "",
                    "description": predicate.get("description") or "",
                    "created_at": predicate.get("created_at") or "",
                }
            )
        bar.update(len(payload.get("content", [])))

        page += 1
        if limit is not None and len(records) >= limit:
            records = records[:limit]
            break
        if page >= payload.get("page", {}).get("total_pages", 0):
            break

    bar.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="target file (default: opo/input/orkg_properties_<today>.json)")
    parser.add_argument("--limit", type=int, help="only fetch the first N properties (for testing)")
    args = parser.parse_args()

    output = args.output or DEFAULT_DIR / f"orkg_properties_{date.today().isoformat()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    records = fetch_predicates(limit=args.limit)
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    # The snapshot is only meaningful together with the moment it was taken: the analysis
    # data and this dump have to describe the same state of a graph that keeps changing.
    meta_path = output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": API_URL,
                "started_at": started_at,
                "finished_at": finished_at,
                "n_properties": len(records),
                "limit": args.limit,
            },
            handle,
            indent=2,
        )

    print(f"{len(records)} properties -> {output}  ({time.time() - started:.1f}s)")
    print(f"snapshot taken {started_at} .. {finished_at}")
    print(f"metadata -> {meta_path}")


if __name__ == "__main__":
    main()
