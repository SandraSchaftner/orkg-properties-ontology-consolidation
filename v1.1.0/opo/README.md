# OPO consolidation pipeline (v1.1.0)

Folds the ~14,000 ORKG predicates into canonical properties, so that consumers can count
property usage without counting the 23 different predicate ids that all mean "method" as 23
different properties.

The output is the Turtle and JSON-LD pair in `ontology/` at the repository root. Everything else
in this folder only exists to produce those files. See [CHANGES.md](CHANGES.md) for the release
notes and for how this pipeline differs from the original one published with the paper.

## Setup

```bash
pip install -r opo/requirements.txt      # heavy: torch + sentence-transformers, ~3 GB
```

The API key is read from `.env` in the repository root (gitignored), so nothing has to be
exported by hand. An already exported `KISTE_API_KEY` takes precedence.

The embedding model (`Qwen/Qwen3-Embedding-8B`, ~14 GB) is downloaded to the Hugging Face
cache on first use and only needed for steps 3 and 4.

## Running

```bash
# 1. snapshot of the current ORKG properties -> opo/input/orkg_properties_<today>.json
python opo/fetch_properties.py

# 2. smoke test on a few hundred properties (minutes, separate run directory)
python opo/opo_consolidation.py --limit 300 --run-dir opo/runs/smoke --end-step 2

# 3. unattended part: lexical dedup, quality filtering, clustering (~4 h)
python opo/opo_consolidation.py --end-step 3

# 4. manual expert review, whenever there is time — needs a terminal,
#    the run refuses to start step 4 without one instead of hanging silently.
#    Commands: largest / find / split / merge / candidates / done
python opo/opo_consolidation.py --start-step 4 --end-step 4

# 5. normalisation and export -> ontology/opo-consolidated-<version>.ttl and .jsonld
python opo/opo_consolidation.py --start-step 5
```

If the result feeds a downstream analysis, take the property snapshot **right after** that
analysis has downloaded its raw data, so the consolidation covers every predicate the data can
contain.

## Layout of a run

Paths are relative to the repository root; the pipeline itself lives in `v1.1.0/opo/`.

```
v1.1.0/opo/
  input/orkg_properties_<date>.json     input snapshot (committed)
  prompts.yaml                          LLM prompts (unchanged from the original)
  runs/<name>/
    cache/*.jsonl                       every single model decision, for resuming (not committed)
    checkpoints/step_*/                 data.csv, mappings.json, stats.json per step
ontology/                               the final artifacts
```

Interrupting is safe at any point: restart the same command and the cached decisions are
reused. Re-running a finished step takes about a second.
