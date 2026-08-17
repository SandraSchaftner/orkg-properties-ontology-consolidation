"""
OPO Consolidation Pipeline (server edition)

Consolidates the crowdsourced ORKG properties ontology into a canonical property set
while keeping backward compatibility through skos:exactMatch mappings.

Derived from `opo_consolidation.py` of
https://github.com/SandraSchaftner/orkg-properties-ontology-consolidation
(Schaftner & Gaedke, 2026). The five-step method is unchanged; what changed is how the
pipeline is executed. See CHANGES.md for the full list and the reasoning.

The five steps:

1.  Lexical Deduplication: resolves case variations and exact string matches.
2.  Semantic Quality Filtering: filters non-predicate labels (objects, topics, noise)
    using LLM-based verification.
3.  Hybrid Semantic Clustering: vector embeddings (local) + LLM cluster validation.
4.  Interactive Expert Review: human-in-the-loop refinement of complex clusters.
5.  Normalization & Canonical Selection: standardises labels and writes the final
    ontology artifacts (JSON-LD, Turtle).

Chat completions run against an OpenAI-compatible server (Kiste); embeddings run
locally, because the server offers no embedding endpoint.

Usage:
    export KISTE_API_KEY=...                     # or put it in .env and export it
    python opo/fetch_properties.py               # snapshot of the current ORKG properties
    python opo/opo_consolidation.py              # full run, steps 1-5
    python opo/opo_consolidation.py --start-step 4        # resume at the manual review
    python opo/opo_consolidation.py --limit 300 --run-dir opo/runs/smoke   # small test run

Dependencies: see opo/requirements.txt
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

HERE = Path(__file__).parent

DEFAULT_API_URL = "https://kiste.informatik.tu-chemnitz.de/v1"
DEFAULT_LLM_MODEL = "Qwen3.6-35B-A3B-MLX-8bit"
DEFAULT_WORKERS = 8  # the server saturates at ~4 req/s; more workers only add latency
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

PROMPTS_FILE = HERE / "prompts.yaml"
DEFAULT_RUN_DIR = HERE / "runs" / "current"

# Agglomerative clustering thresholds (cosine distance) — unchanged from the original run.
THRESHOLD_LOOSE = 0.25
THRESHOLD_STRICT = 0.15

GARBAGE_LENGTH_THRESHOLD = 3
LINK_URI = "https://orkg.org/property/P41267"

ONTOLOGY_URI = "https://w3id.org/orkg-properties-ontology-consolidated"
ONTOLOGY_VERSION = "1.1.0"
# The base URI names the ontology, the version IRI names this release. Without it two
# releases are indistinguishable as RDF — they differ only in a versionInfo literal and in
# the file name. `priorVersion` links back to what this release supersedes.
VERSION_IRI = f"{ONTOLOGY_URI}/{ONTOLOGY_VERSION}"
PRIOR_VERSION_IRI = f"{ONTOLOGY_URI}/1.0.0"

# JSON schemas for the structured LLM answers.
SCHEMA_SELECTED_URI = {
    "type": "object",
    "properties": {"selected_uri": {"type": "string"}},
    "required": ["selected_uri"],
    "additionalProperties": False,
}
SCHEMA_IS_VALID = {
    "type": "object",
    "properties": {"is_valid": {"type": "boolean"}},
    "required": ["is_valid"],
    "additionalProperties": False,
}
SCHEMA_DECISION = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["KEEP", "BREAK"]}},
    "required": ["decision"],
    "additionalProperties": False,
}
SCHEMA_CLEAN_LABEL = {
    "type": "object",
    "properties": {"clean_label": {"type": "string"}},
    "required": ["clean_label"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------------------
# Infrastructure: LLM client, disk cache, parallel execution
# --------------------------------------------------------------------------------------


class LLMClient:
    """Minimal client for an OpenAI-compatible /chat/completions endpoint.

    Uses `requests` directly instead of the `openai` package: the pipeline only needs one
    endpoint with structured outputs, and this keeps the dependency list short. Structured
    outputs are enforced server-side (`response_format.json_schema` with `strict: true`),
    which also suppresses the model's reasoning tokens and makes each call ~25x cheaper.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 600, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self.failures = 0
        self._lock = threading.Lock()

    def ask(self, system_prompt: str, user_prompt: str, schema: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """One structured query. Returns {} if the call fails permanently (counted in `failures`)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        }
        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(strip_thinking(content))
            except Exception as error:  # noqa: BLE001 - network/JSON/shape errors are all retryable
                if attempt == self.retries - 1:
                    with self._lock:
                        self.failures += 1
                    print(f"  LLM call failed after {self.retries} attempts: {error}")
                    return {}
                time.sleep(2**attempt)
        return {}

    def check(self) -> None:
        """Fail fast if the server or the model is not reachable."""
        answer = self.ask("You answer with JSON.", 'Is "accuracy" a valid property?', SCHEMA_IS_VALID, "healthcheck")
        if "is_valid" not in answer:
            raise RuntimeError(
                f"LLM server not usable: {self.base_url} / {self.model}. "
                "Check KISTE_API_KEY, the model id (GET /v1/models) and the network."
            )
        print(f"LLM server OK: {self.model} @ {self.base_url}")


class DecisionCache:
    """Append-only JSONL cache of individual LLM decisions.

    The original pipeline only checkpointed *between* steps, so a crash three hours into
    step 2 lost everything. Here every decision is written out as soon as it is made, and
    a restart skips whatever is already known.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, Any] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self._entries[record["key"]] = record["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a truncated last line from a hard crash

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str, default: Any = None) -> Any:
        return self._entries.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._entries[key] = value
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "value": value}, default=str) + "\n")
                handle.flush()


def parallel_map(
    work: Callable[[Any], Any],
    items: List[Any],
    workers: int,
    desc: str,
) -> List[Any]:
    """Run `work` over `items` concurrently, preserving input order."""
    if not items:
        return []
    if workers <= 1:
        return [work(item) for item in tqdm(items, desc=desc)]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, item): index for index, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc) as bar:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                bar.update(1)
    return results


def cached_parallel(
    items: List[Any],
    key_of: Callable[[Any], str],
    work: Callable[[Any], Any],
    cache: DecisionCache,
    workers: int,
    desc: str,
) -> Dict[str, Any]:
    """Compute `work(item)` for every item whose key is not cached yet; return key -> value."""
    pending = [item for item in items if key_of(item) not in cache]
    known = len(items) - len(pending)
    if known:
        print(f"  {known} of {len(items)} already cached, {len(pending)} to do")

    def run(item):
        value = work(item)
        cache.set(key_of(item), value)
        return value

    parallel_map(run, pending, workers, desc)
    return {key_of(item): cache.get(key_of(item)) for item in items}


def load_env(path: Path = HERE.parent / ".env") -> None:
    """Read KEY=value lines from .env into the environment (existing values win).

    Small enough not to warrant a python-dotenv dependency.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def strip_thinking(text: str) -> str:
    """Remove reasoning traces if a model emits them despite structured output."""
    return re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def hash_key(*parts: str) -> str:
    """Stable cache key for composite inputs (e.g. all labels of a cluster)."""
    joined = "\x00".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Embeddings (local)
# --------------------------------------------------------------------------------------


def compute_embeddings(labels: List[str], model_name: str, cache_file: Path) -> np.ndarray:
    """Embed labels with a local sentence-transformers model, cached on disk.

    The inference server has no embedding endpoint, so this part stays local. Loading the
    8B model takes a few minutes and a lot of memory; the cache makes sure it happens once
    per label set, not once per run.
    """
    fingerprint = hash_key(model_name, *labels)
    if cache_file.exists():
        stored = np.load(cache_file, allow_pickle=False)
        if "fingerprint" in stored.files and str(stored["fingerprint"]) == fingerprint:
            print(f"  embeddings loaded from cache ({cache_file.name})")
            return stored["embeddings"]
        print("  embedding cache does not match this label set, recomputing")

    import torch  # imported lazily: steps 1, 2 and 5 do not need it
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  loading embedding model {model_name} on {device} ...")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    embeddings = model.encode(labels, normalize_embeddings=True, show_progress_bar=True)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, embeddings=embeddings, fingerprint=np.array(fingerprint))
    return embeddings


def step3_embeddings(df: pd.DataFrame, run: "RunContext") -> Tuple[np.ndarray, Dict[str, int]]:
    """Embeddings of the clustered labels, reusing step 3's cache.

    The link property is not part of the clustering, so it is excluded here too — that
    keeps the label set identical to step 3's and the cache hits instead of reloading a
    14 GB model.
    """
    clustered = df[df["uri"] != LINK_URI]
    labels = clustered["label"].astype(str).tolist()
    matrix = compute_embeddings(labels, run.embedding_model, run.dir / "cache" / "embeddings_step3.npz")
    index_of = {label.strip().lower(): position for position, label in enumerate(labels)}
    return matrix, index_of


def encode_terms(terms: List[str], run: "RunContext", matrix: np.ndarray, index_of: Dict[str, int]) -> np.ndarray:
    """Vectors for arbitrary center terms.

    Terms that are existing labels are looked up in the cached matrix; only genuinely new
    wording requires loading the embedding model.
    """
    known = [term for term in terms if term.strip().lower() in index_of]
    unknown = [term for term in terms if term.strip().lower() not in index_of]

    encoded: Dict[str, np.ndarray] = {
        term: matrix[index_of[term.strip().lower()]] for term in known
    }
    if unknown:
        print(f"  encoding {len(unknown)} new term(s), loading the embedding model ...")
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(run.embedding_model, trust_remote_code=True, device=device)
        vectors = model.encode(unknown, normalize_embeddings=True)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        encoded.update(dict(zip(unknown, vectors)))

    return np.vstack([encoded[term] for term in terms]).astype(np.float32)


def merge_candidates(
    df: pd.DataFrame, matrix: np.ndarray, top_n: int = 25, min_similarity: float = 0.75
) -> List[Dict[str, Any]]:
    """Cluster pairs whose centroids are closest — the fragments worth merging back.

    Step 3 splits aggressively (the strict threshold plus the model's tendency to break
    clusters), which is safer than merging too much but leaves the same concept spread over
    several clusters. This ranks the pairs so the review does not have to find them by hand.
    """
    clustered = df[df["uri"] != LINK_URI].reset_index(drop=True)
    if len(clustered) != len(matrix):
        raise ValueError("embedding matrix does not match the clustered rows")

    cluster_ids = sorted(clustered["cluster_id"].unique())
    centroids = np.vstack(
        [matrix[clustered.index[clustered["cluster_id"] == cluster_id]].mean(axis=0) for cluster_id in cluster_ids]
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids = centroids / norms

    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -1.0)

    pairs = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if similarity[i, j] >= min_similarity:
                pairs.append((float(similarity[i, j]), cluster_ids[i], cluster_ids[j]))
    pairs.sort(reverse=True)

    labels_of = clustered.groupby("cluster_id")["label"].apply(list).to_dict()
    return [
        {
            "similarity": round(score, 3),
            "cluster_a": int(a),
            "cluster_b": int(b),
            "labels_a": sorted(map(str, labels_of[a])),
            "labels_b": sorted(map(str, labels_of[b])),
        }
        for score, a, b in pairs[:top_n]
    ]


def agglomerative(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster normalised embeddings by cosine distance."""
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
    )
    return model.fit_predict(embeddings)


# --------------------------------------------------------------------------------------
# Mappings and checkpoints
# --------------------------------------------------------------------------------------


def add_mapping(
    mappings: List[Dict[str, Any]],
    old_row: Union[pd.Series, Dict[str, Any]],
    new_uri: str,
    relation: str = "skos:exactMatch",
) -> None:
    """Record that an original property is represented by a canonical one."""
    if old_row["uri"] == new_uri:
        return
    mappings.append(
        {
            "original_uri": old_row["uri"],
            "original_id": old_row.get("id", "Unknown"),
            "mapped_to_uri": new_uri,
            "relation": relation,
            "original_label": old_row.get("label", ""),
            "created_at": old_row.get("created_at", ""),
            "description": old_row.get("description", ""),
        }
    )


def generate_cluster_report(df: pd.DataFrame, filename: Path) -> None:
    """Human-readable overview of the current clusters, used for the step 4 review."""
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(f"CLUSTER REPORT - Generated at {datetime.now()}\n")
        handle.write("=" * 60 + "\n\n")
        if "cluster_id" not in df.columns:
            return
        for cluster_id in df["cluster_id"].value_counts().index:
            subset = df[df["cluster_id"] == cluster_id]
            handle.write(f"=== CLUSTER {cluster_id} (Size: {len(subset)}) ===\n")
            for label in sorted(subset["label"].unique().tolist()):
                handle.write(f"- {label}\n")
            handle.write("\n")


def save_checkpoint(run_dir: Path, step_name: str, df: pd.DataFrame, mappings: List[Dict], stats: Dict) -> None:
    """Persist data, mappings and stats of a completed step."""
    folder = run_dir / "checkpoints" / step_name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\nCheckpoint '{step_name}' -> {folder}")

    df.to_csv(folder / "data.csv", index=False)

    def convert(value):
        return int(value) if isinstance(value, (np.integer,)) else value

    with open(folder / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=4, default=convert)
    with open(folder / "mappings.json", "w", encoding="utf-8") as handle:
        json.dump(mappings, handle, indent=4, default=convert)

    with open(folder / "mappings.ttl", "w", encoding="utf-8") as handle:
        handle.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        handle.write("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n")
        handle.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n")
        for mapping in mappings:
            safe_label = json.dumps(str(mapping["original_label"]))
            handle.write(f"<{mapping['original_uri']}> skos:exactMatch <{mapping['mapped_to_uri']}> ;\n")
            handle.write(f"    owl:equivalentProperty <{mapping['mapped_to_uri']}> ;\n")
            handle.write(f"    rdfs:label {safe_label} .\n")

    if "cluster_id" in df.columns:
        generate_cluster_report(df, folder / "cluster_overview.txt")


def load_checkpoint(run_dir: Path, step_name: str) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Restore data, mappings and stats of a previously completed step."""
    folder = run_dir / "checkpoints" / step_name
    df = pd.read_csv(folder / "data.csv").fillna("")
    mappings, stats = [], {}
    if (folder / "mappings.json").exists():
        mappings = json.loads((folder / "mappings.json").read_text(encoding="utf-8"))
        print(f"  restored {len(mappings)} mappings from {step_name}")
    else:
        print(f"  WARNING: no mappings.json in {step_name}, provenance may be lost")
    if (folder / "stats.json").exists():
        stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
    return df, mappings, stats


# --------------------------------------------------------------------------------------
# Step 1: Lexical deduplication
# --------------------------------------------------------------------------------------


def step_1_lexical(
    df: pd.DataFrame, llm: LLMClient, prompts: Dict, mappings: List[Dict], stats: Dict, run: "RunContext"
) -> pd.DataFrame:
    """Merge properties whose labels are identical apart from case and whitespace."""
    print("\nSTEP 1: Lexical Deduplication")
    started = time.time()

    df = df.copy()
    df["label_lower"] = df["label"].astype(str).str.lower().str.strip()
    groups = [(label, group) for label, group in df.groupby("label_lower")]
    groups.sort(key=lambda item: item[0])  # deterministic order
    ambiguous = [(label, group) for label, group in groups if group["uri"].nunique() > 1]

    stats["step_1"] = {"total": len(df), "groups": len(groups), "ambiguous_groups": len(ambiguous), "removed": 0}
    cache = run.cache("step_1_disambiguation")
    block = prompts["step_1_disambiguation"]

    def decide(item) -> str:
        label, group = item
        candidates = group.to_dict("records")
        candidates_json = json.dumps(
            [{"uri": c["uri"], "id": c["id"], "desc": c.get("description", "")} for c in candidates]
        )
        answer = llm.ask(
            block["system"],
            block["user_template"].format(label=label, candidates_json=candidates_json),
            SCHEMA_SELECTED_URI,
            "disambiguation_schema",
        )
        selected = answer.get("selected_uri")
        valid = {c["uri"] for c in candidates}
        return selected if selected in valid else candidates[0]["uri"]

    winners = cached_parallel(ambiguous, lambda item: item[0], decide, cache, run.workers, "Lexical disambiguation")

    kept_rows = []
    for label, group in groups:
        candidates = group.to_dict("records")
        if len(candidates) == 1:
            kept_rows.append(candidates[0])
            continue

        winner_uri = winners.get(label) or candidates[0]["uri"]
        winner_kept = False
        for candidate in candidates:
            if candidate["uri"] == winner_uri and not winner_kept:
                kept_rows.append(candidate)
                winner_kept = True
            else:
                if candidate["uri"] != winner_uri:
                    add_mapping(mappings, candidate, winner_uri)
                stats["step_1"]["removed"] += 1

    stats["step_1"]["duration_s"] = round(time.time() - started, 1)
    result = pd.DataFrame(kept_rows).drop(columns=["label_lower"])
    print(f"-> {len(df)} properties in, {len(result)} out ({stats['step_1']['removed']} merged)")
    return result


# --------------------------------------------------------------------------------------
# Step 2: Semantic quality filtering
# --------------------------------------------------------------------------------------


def step_2_quality(
    df: pd.DataFrame, llm: LLMClient, prompts: Dict, mappings: List[Dict], stats: Dict, run: "RunContext"
) -> pd.DataFrame:
    """Drop labels that are not predicates (objects, topics, sentences, noise).

    Rejected properties are not deleted: they are mapped onto the generic link property
    so that existing statements remain resolvable.
    """
    print("\nSTEP 2: Quality Control")
    started = time.time()

    stats["step_2"] = {"checked": len(df), "rejected_regex": 0, "rejected_llm": 0}
    cache = run.cache("step_2_quality")
    long_block = prompts["step_2_long_label_check"]
    semantic_block = prompts["step_2_semantic_check"]

    def judge(row: Dict[str, Any]) -> Dict[str, Any]:
        label = str(row["label"]).strip()
        description = str(row.get("description", "") or "")

        if len(label.split()) > GARBAGE_LENGTH_THRESHOLD:
            answer = llm.ask(
                long_block["system"],
                long_block["user_template"].format(label=label, description=description),
                SCHEMA_IS_VALID,
                "qc_long",
            )
            if not answer.get("is_valid", True):
                return {"is_valid": False, "reason": "long_label"}

        answer = llm.ask(
            semantic_block["system"],
            semantic_block["user_template"].format(label=label, description=description),
            SCHEMA_IS_VALID,
            "qc_sem",
        )
        if not answer.get("is_valid", True):
            return {"is_valid": False, "reason": "semantic"}
        return {"is_valid": True, "reason": "accepted"}

    # The regex filter needs no model, so it runs first and shrinks the LLM workload.
    rows = df.to_dict("records")
    to_check, regex_rejected = [], set()
    for row in rows:
        label = str(row["label"]).strip()
        if row["uri"] == LINK_URI:
            continue
        if re.match(r"^(orkg:)?(P|R)\d+(\s.*)?$", label, re.IGNORECASE):
            regex_rejected.add(row["uri"])
        else:
            to_check.append(row)

    verdicts = cached_parallel(to_check, lambda row: row["uri"], judge, cache, run.workers, "Quality filtering")

    accepted_rows = []
    for row in rows:
        if row["uri"] == LINK_URI:
            accepted_rows.append(row)
            continue
        if row["uri"] in regex_rejected:
            add_mapping(mappings, row, LINK_URI)
            stats["step_2"]["rejected_regex"] += 1
            continue
        verdict = verdicts.get(row["uri"], {"is_valid": True})
        if verdict.get("is_valid", True):
            accepted_rows.append(row)
        else:
            add_mapping(mappings, row, LINK_URI)
            stats["step_2"]["rejected_llm"] += 1

    stats["step_2"]["duration_s"] = round(time.time() - started, 1)
    print(
        f"-> {len(accepted_rows)} accepted, "
        f"{stats['step_2']['rejected_regex']} rejected by regex, "
        f"{stats['step_2']['rejected_llm']} rejected by the model"
    )
    return pd.DataFrame(accepted_rows)


# --------------------------------------------------------------------------------------
# Step 3: Hybrid semantic clustering
# --------------------------------------------------------------------------------------


def step_3_semantic(
    df: pd.DataFrame, llm: LLMClient, prompts: Dict, mappings: List[Dict], stats: Dict, run: "RunContext"
) -> pd.DataFrame:
    """Group semantically equivalent properties.

    Phase A clusters loosely, phase B asks the model whether a cluster mixes distinct
    concepts, phase C re-clusters the rejected ones with a stricter threshold.
    """
    print(f"\nSTEP 3: Hybrid Semantic Clustering (loose {THRESHOLD_LOOSE} -> LLM -> strict {THRESHOLD_STRICT})")
    started = time.time()

    # The link property collects everything step 2 rejected, so it has to stay recognisable
    # as a terminal node. Carrying a label like any other property, it would otherwise be
    # clustered and merged along with the rest (in v1.0.0 it joined the cluster around
    # "links", P41267 -> P15325), and every rejected property would then resolve to a
    # legitimate one. It is set aside here and re-added with a cluster of its own, so
    # steps 4 and 5 cannot merge it away.
    protected = df[df["uri"] == LINK_URI]
    df = df[df["uri"] != LINK_URI]
    if len(protected):
        print(f"  {LINK_URI.rsplit('/', 1)[-1]} excluded from clustering (link property)")

    df = df.reset_index(drop=True)
    labels = df["label"].astype(str).tolist()
    embeddings = compute_embeddings(labels, run.embedding_model, run.dir / "cache" / "embeddings_step3.npz")

    print(f"  phase A: loose clustering at {THRESHOLD_LOOSE}")
    df["temp_cluster"] = agglomerative(embeddings, THRESHOLD_LOOSE)

    cluster_sizes = df["temp_cluster"].value_counts()
    sorted_clusters = cluster_sizes.index.tolist()
    multi = [cluster_id for cluster_id in sorted_clusters if cluster_sizes[cluster_id] > 1]

    stats["step_3"] = {
        "loose_clusters": int(df["temp_cluster"].nunique()),
        "llm_kept": 0,
        "llm_broken": 0,
        "decisions": [],
    }
    cache = run.cache("step_3_validation")
    block = prompts["step_3_cluster_validation"]

    def cluster_labels(cluster_id) -> List[str]:
        return df.loc[df["temp_cluster"] == cluster_id, "label"].unique().tolist()

    def validate(cluster_id) -> str:
        answer = llm.ask(
            block["system"],
            block["user_template"].format(labels=json.dumps(cluster_labels(cluster_id))),
            SCHEMA_DECISION,
            "cluster_val",
        )
        return answer.get("decision", "BREAK")

    print(f"  phase B: validating {len(multi)} multi-member clusters")
    decisions = cached_parallel(
        multi,
        lambda cluster_id: hash_key(*sorted(cluster_labels(cluster_id))),
        validate,
        cache,
        run.workers,
        "Validating clusters",
    )

    # Cluster ids are assigned sequentially in a deterministic order, so the result does
    # not depend on the order in which the parallel validations happened to finish.
    print("  phase C: strict re-clustering of rejected clusters")
    final_cluster_map: Dict[int, int] = {}
    next_id = 0
    for cluster_id in sorted_clusters:
        indices = df.index[df["temp_cluster"] == cluster_id].tolist()
        entry = {"loose_cluster_id": int(cluster_id), "labels": cluster_labels(cluster_id), "strict_subclusters": []}

        if len(indices) == 1:
            final_cluster_map[indices[0]] = next_id
            next_id += 1
            entry["decision"] = "SINGLETON"
            stats["step_3"]["decisions"].append(entry)
            continue

        decision = decisions.get(hash_key(*sorted(cluster_labels(cluster_id))), "BREAK")
        entry["decision"] = decision

        if decision == "KEEP":
            for index in indices:
                final_cluster_map[index] = next_id
            next_id += 1
            stats["step_3"]["llm_kept"] += 1
        else:
            stats["step_3"]["llm_broken"] += 1
            strict_labels = agglomerative(embeddings[indices], THRESHOLD_STRICT)
            for sub_id in np.unique(strict_labels):
                members = [indices[i] for i in range(len(indices)) if strict_labels[i] == sub_id]
                entry["strict_subclusters"].append(
                    {"sub_id": int(sub_id), "labels": df.loc[members, "label"].tolist()}
                )
                for index in members:
                    final_cluster_map[index] = next_id
                next_id += 1

        stats["step_3"]["decisions"].append(entry)

    df["cluster_id"] = [final_cluster_map[index] for index in range(len(df))]
    df = df.drop(columns=["temp_cluster"])

    if len(protected):
        protected = protected.copy()
        protected["cluster_id"] = next_id  # a cluster of its own, cannot be merged away
        df = pd.concat([df, protected], ignore_index=True)

    stats["step_3"]["duration_s"] = round(time.time() - started, 1)
    print(
        f"-> kept {stats['step_3']['llm_kept']}, broke {stats['step_3']['llm_broken']} clusters"
        f"  |  {df['cluster_id'].nunique()} clusters in total"
    )
    return df


# --------------------------------------------------------------------------------------
# Step 4: Interactive expert review + merge
# --------------------------------------------------------------------------------------


def step_4_interactive(
    df: pd.DataFrame, llm: LLMClient, prompts: Dict, mappings: List[Dict], stats: Dict, run: "RunContext"
) -> pd.DataFrame:
    """Manual cluster review (split and merge), then reduce each cluster to one property."""
    print("\nSTEP 4: Interactive Review & Merge")
    started = time.time()
    stats.setdefault("step_4", {"interactive_splits": [], "interactive_merges": [], "merged": 0})
    stats["step_4"].setdefault("interactive_merges", [])

    report = run.dir / "checkpoints" / "step_3_clustered" / "cluster_overview.txt"
    print(f"TIP: open '{report}' to find the clusters that need a look.")

    df = df.reset_index(drop=True)
    matrix, index_of = step3_embeddings(df, run)

    def show(cluster_id: int) -> bool:
        subset = df[df["cluster_id"] == cluster_id]
        if subset.empty:
            print(f"  no cluster {cluster_id}")
            return False
        print(f"\nCluster {cluster_id} ({len(subset)} labels):")
        for label in sorted(subset["label"].astype(str), key=str.lower):
            print(f"    {label}")
        return True

    HELP = """
Commands:
  <id>                    show a cluster
  largest [n]             list the n largest clusters (default 10)
  split <id>              split a cluster into new center terms
  merge <id> <id> [...]   merge clusters into the first one
  candidates [n]          rank cluster pairs that look like the same concept
  find <text>             find clusters containing a label
  done                    finish the review and continue with the merge phase
"""
    print(HELP)

    while True:
        try:
            answer = input("\nreview> ").strip()
        except EOFError:
            break
        if not answer:
            continue
        parts = answer.split()
        command, args = parts[0].lower(), parts[1:]

        if command in {"done", "no", "n", "exit", "quit"}:
            break

        if command in {"help", "?"}:
            print(HELP)

        elif command == "largest":
            count = int(args[0]) if args and args[0].isdigit() else 10
            print(df["cluster_id"].value_counts().head(count).to_string())

        elif command == "find":
            needle = " ".join(args).lower()
            hits = df[df["label"].astype(str).str.lower().str.contains(needle, regex=False)]
            for row in hits.head(30).itertuples():
                print(f"  cluster {row.cluster_id:>6}  {row.label}")
            print(f"  ({len(hits)} matches)")

        elif command == "candidates":
            count = int(args[0]) if args and args[0].isdigit() else 15
            for candidate in merge_candidates(df, matrix, top_n=count):
                print(f"\n  similarity {candidate['similarity']}  ->  merge {candidate['cluster_a']} {candidate['cluster_b']}")
                print(f"    {candidate['cluster_a']}: {', '.join(candidate['labels_a'][:8])}")
                print(f"    {candidate['cluster_b']}: {', '.join(candidate['labels_b'][:8])}")

        elif command == "merge":
            try:
                ids = [int(a) for a in args]
            except ValueError:
                print("  usage: merge <id> <id> [...]")
                continue
            if len(ids) < 2:
                print("  need at least two cluster ids")
                continue
            existing = [i for i in ids if (df["cluster_id"] == i).any()]
            if len(existing) < 2:
                print("  at least two of those clusters do not exist")
                continue
            # The link property collects the rejected ones and must not be merged into a
            # real property (see the note in step 3).
            if df[df["cluster_id"].isin(existing)]["uri"].eq(LINK_URI).any():
                print("  refusing: one of these clusters holds the link property")
                continue
            target, sources = existing[0], existing[1:]
            moved = df[df["cluster_id"].isin(sources)]
            stats["step_4"]["interactive_merges"].append(
                {
                    "target_cluster": target,
                    "source_clusters": sources,
                    "labels": moved["label"].astype(str).tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            df.loc[df["cluster_id"].isin(sources), "cluster_id"] = target
            print(f"  merged {len(moved)} labels from {sources} into {target}")
            show(target)

        elif command == "split" or command.isdigit():
            if command.isdigit():
                if show(int(command)):
                    continue
                continue
            if not args or not args[0].isdigit():
                print("  usage: split <id>")
                continue
            cluster_id = int(args[0])
            subset = df[df["cluster_id"] == cluster_id]
            if subset.empty:
                print(f"  no cluster {cluster_id}")
                continue
            show(cluster_id)
            centers_input = input("New center terms, comma separated (empty to cancel): ").strip()
            if not centers_input:
                continue
            centers = [center.strip() for center in centers_input.split(",") if center.strip()]
            if len(centers) < 2:
                print("  need at least two center terms")
                continue

            center_vectors = encode_terms(centers, run, matrix, index_of)
            stats["step_4"]["interactive_splits"].append(
                {
                    "cluster_id": cluster_id,
                    "centers_input": centers_input,
                    "original_labels": subset["label"].astype(str).tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            base_id = int(df["cluster_id"].max()) + 1
            for index in subset.index:
                label = str(df.at[index, "label"]).strip().lower()
                position = index_of.get(label)
                if position is None:  # the link property has no embedding
                    continue
                scores = center_vectors @ matrix[position].astype(np.float32)
                df.at[index, "cluster_id"] = base_id + int(np.argmax(scores))
            print(f"  split into {len(centers)} clusters: {base_id}..{base_id + len(centers) - 1}")
            for offset in range(len(centers)):
                show(base_id + offset)

        else:
            print("  unknown command — type 'help'")

    # Merge phase: one canonical property per cluster.
    cache = run.cache("step_4_merge")
    block = prompts["step_4_cluster_selection"]
    clusters = [(cluster_id, group) for cluster_id, group in df.groupby("cluster_id") if len(group) > 1]

    def select(item) -> str:
        _, group = item
        candidates = group.to_dict("records")
        candidates_json = json.dumps([{"uri": c["uri"], "label": c["label"]} for c in candidates])
        answer = llm.ask(
            block["system"],
            block["user_template"].format(candidates_json=candidates_json),
            SCHEMA_SELECTED_URI,
            "merge",
        )
        selected = answer.get("selected_uri")
        valid = {c["uri"] for c in candidates}
        if selected in valid:
            return selected
        return sorted(candidates, key=lambda row: len(str(row["label"])))[0]["uri"]

    winners = cached_parallel(
        clusters,
        lambda item: hash_key(*sorted(str(uri) for uri in item[1]["uri"])),
        select,
        cache,
        run.workers,
        "Merging clusters",
    )

    final_rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        candidates = group.to_dict("records")
        if len(candidates) == 1:
            final_rows.append(candidates[0])
            continue
        key = hash_key(*sorted(str(row["uri"]) for row in candidates))
        winner_uri = winners.get(key) or sorted(candidates, key=lambda row: len(str(row["label"])))[0]["uri"]
        for candidate in candidates:
            if candidate["uri"] == winner_uri:
                final_rows.append(candidate)
            else:
                add_mapping(mappings, candidate, winner_uri)
                stats["step_4"]["merged"] += 1

    stats["step_4"]["duration_s"] = round(time.time() - started, 1)
    print(f"-> {len(final_rows)} properties after merging ({stats['step_4']['merged']} merged away)")
    return pd.DataFrame(final_rows)


# --------------------------------------------------------------------------------------
# Step 5: Normalization and final deduplication
# --------------------------------------------------------------------------------------


def step_5_normalization(
    df: pd.DataFrame, llm: LLMClient, prompts: Dict, mappings: List[Dict], stats: Dict, run: "RunContext"
) -> pd.DataFrame:
    """Normalise labels to ORKG best practice, then merge labels that collide afterwards."""
    print("\nSTEP 5: Label Normalization & Final Deduplication")
    started = time.time()
    stats["step_5"] = {"normalized": 0, "final_duplicates_merged": 0}

    df = df.reset_index(drop=True)
    df["clean_label_temp"] = df["label"].astype(str)

    norm_cache = run.cache("step_5_normalization")
    norm_block = prompts["step_5_normalization"]

    def normalize(label: str) -> str:
        # Already a lowercase single word: nothing for the model to do. Underscores and
        # colons do not count as "single word" — `has_method` and `paper:venue` look like
        # one lowercase token but are exactly what normalisation is supposed to clean up.
        if label.islower() and not any(char in label for char in " _:") and len(label) > 2:
            return label
        answer = llm.ask(
            norm_block["system"], norm_block["user_template"].format(label=label), SCHEMA_CLEAN_LABEL, "norm"
        )
        return answer.get("clean_label", label) or label

    print("  phase 1: normalising labels")
    unique_labels = sorted(df["label"].astype(str).unique())
    clean_by_label = cached_parallel(
        unique_labels, lambda label: label, normalize, norm_cache, run.workers, "Normalizing"
    )
    for index, row in df.iterrows():
        if row["uri"] == LINK_URI:
            continue  # the link property keeps its label, see the note in step 3
        clean = clean_by_label.get(str(row["label"]), str(row["label"]))
        if clean != str(row["label"]):
            df.at[index, "clean_label_temp"] = clean
            stats["step_5"]["normalized"] += 1

    print("  phase 2: resolving label collisions")
    dedup_cache = run.cache("step_5_dedup")
    dedup_block = prompts["step_5_final_dedup"]
    collisions = [
        (label, group) for label, group in df.groupby("clean_label_temp") if len(group) > 1
    ]

    def resolve(item) -> str:
        label, group = item
        candidates = group.to_dict("records")
        candidates_json = json.dumps(
            [{"uri": c["uri"], "id": c.get("id", ""), "desc": c.get("description", "")} for c in candidates]
        )
        answer = llm.ask(
            dedup_block["system"],
            dedup_block["user_template"].format(label=label, candidates_json=candidates_json),
            SCHEMA_SELECTED_URI,
            "final_dedup",
        )
        selected = answer.get("selected_uri")
        valid = {c["uri"] for c in candidates}
        if selected in valid:
            return selected
        fallback = sorted(candidates, key=lambda row: (len(str(row.get("id", ""))), str(row["uri"])))
        return fallback[0]["uri"]

    winners = cached_parallel(
        collisions, lambda item: str(item[0]), resolve, dedup_cache, run.workers, "Deduplicating"
    )

    drop_indices = []
    for label, group in collisions:
        candidates = group.to_dict("records")
        # The link property takes no part in collisions: merging it away would hide the
        # rejected properties, and merging a real property into it would silently declare
        # that property rejected. It simply keeps its label, even if that leaves a
        # duplicate — a visible duplicate is better than a wrong merge in either direction.
        if any(candidate["uri"] == LINK_URI for candidate in candidates):
            candidates = [c for c in candidates if c["uri"] != LINK_URI]
            if len(candidates) < 2:
                continue
        winner_uri = winners.get(str(label)) or candidates[0]["uri"]
        if winner_uri == LINK_URI:
            winner_uri = candidates[0]["uri"]
        for candidate in candidates:
            if candidate["uri"] != winner_uri:
                add_mapping(mappings, candidate, winner_uri)
                drop_indices.extend(df.index[df["uri"] == candidate["uri"]].tolist())
                stats["step_5"]["final_duplicates_merged"] += 1

    df = df.drop(index=sorted(set(drop_indices)))
    df["label"] = df["clean_label_temp"]
    df = df.drop(columns=["clean_label_temp"])

    stats["step_5"]["duration_s"] = round(time.time() - started, 1)
    print(f"-> {len(df)} canonical properties")
    return df


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------


def export_ontology(
    df: pd.DataFrame, mappings: List[Dict], run: "RunContext", stats: Optional[Dict] = None
) -> Tuple[Path, Path]:
    """Write the consolidated ontology as JSON-LD and Turtle."""
    from rdflib import OWL, RDF, RDFS, XSD, Graph, Namespace, URIRef

    print("\nWriting ontology artifacts ...")
    export_columns = [c for c in ["uri", "id", "label", "description", "created_at"] if c in df.columns]
    records = df[export_columns].to_dict(orient="records")
    stamp = date_stamp()

    # `created` stays a plain date, as in v1.0.0, so the two versions stay comparable.
    # `issued` carries the exact moment this file was written, and `source` names the
    # property snapshot it was derived from — together they pin down which state of ORKG
    # this ontology describes.
    issued = datetime.now().astimezone().isoformat(timespec="seconds")
    source_file = Path((stats or {}).get("run", {}).get("input_file", "")).name or "unknown"

    graph_entries: List[Dict[str, Any]] = [
        {
            "@id": ONTOLOGY_URI,
            "@type": "owl:Ontology",
            "dcterms:title": "ORKG Properties Ontology Consolidated (OPO-Consolidated)",
            "dcterms:description": "A consolidated ontology of research properties derived from ORKG.",
            "dcterms:creator": "OPO Consolidation Pipeline",
            "dcterms:created": {"@value": stamp, "@type": "xsd:date"},
            "dcterms:issued": {"@value": issued, "@type": "xsd:dateTime"},
            "dcterms:source": source_file,
            "owl:versionInfo": ONTOLOGY_VERSION,
            "owl:versionIRI": {"@id": VERSION_IRI},
            "owl:priorVersion": {"@id": PRIOR_VERSION_IRI},
        }
    ]
    for row in records:
        graph_entries.append(
            {
                "@id": row["uri"],
                "@type": "rdf:Property",
                "label": row.get("label", ""),
                "skos:prefLabel": row.get("label", ""),
                "description": row.get("description", ""),
                "id": row.get("id", ""),
                "created_at": {"@value": row.get("created_at", ""), "@type": "xsd:dateTime"},
                "isDefinedBy": ONTOLOGY_URI,
            }
        )
    for mapping in mappings:
        graph_entries.append(
            {
                "@id": mapping["original_uri"],
                "@type": "rdf:Property",
                "skos:exactMatch": {"@id": mapping["mapped_to_uri"]},
                "owl:equivalentProperty": {"@id": mapping["mapped_to_uri"]},
                "rdfs:label": mapping.get("original_label", ""),
            }
        )

    json_ld = {
        "@context": {
            "orkg": "https://orkg.org/property/",
            "opo": f"{ONTOLOGY_URI}#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "dcterms": "http://purl.org/dc/terms/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "id": "dcterms:identifier",
            "isDefinedBy": {"@id": "rdfs:isDefinedBy", "@type": "@id"},
            "label": {"@id": "rdfs:label", "@language": "en"},
            "skos:prefLabel": {"@id": "skos:prefLabel", "@language": "en"},
            "description": {"@id": "rdfs:comment", "@language": "en"},
            "created_at": "dcterms:created",
        },
        "@graph": graph_entries,
    }

    run.output_dir.mkdir(parents=True, exist_ok=True)
    jsonld_path = run.output_dir / f"opo-consolidated-{ONTOLOGY_VERSION}_{stamp}.jsonld"
    with open(jsonld_path, "w", encoding="utf-8") as handle:
        json.dump(json_ld, handle, indent=2)

    graph = Graph()
    graph.parse(data=json.dumps(json_ld), format="json-ld")

    DCTERMS = Namespace("http://purl.org/dc/terms/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    graph.bind("orkg", Namespace("https://orkg.org/property/"))
    graph.bind("opo", Namespace(f"{ONTOLOGY_URI}#"))
    graph.bind("dcterms", DCTERMS)
    graph.bind("skos", SKOS)
    graph.bind("owl", OWL)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)

    subject = URIRef(ONTOLOGY_URI)
    header_values = {
        "title": graph.value(subject, DCTERMS.title),
        "description": graph.value(subject, DCTERMS.description),
        "creator": graph.value(subject, DCTERMS.creator),
        "created": graph.value(subject, DCTERMS.created),
        "version": graph.value(subject, OWL.versionInfo),
    }
    graph.remove((subject, None, None))
    turtle = graph.serialize(format="turtle")

    header = [
        "",
        f"<{ONTOLOGY_URI}> a owl:Ontology ;",
        f'    dcterms:title "{header_values["title"]}"@en ;',
        f'    dcterms:description "{header_values["description"]}"@en ;',
        f'    dcterms:creator "{header_values["creator"]}" ;',
        f'    dcterms:created "{header_values["created"]}"^^xsd:date ;',
        f'    dcterms:issued "{issued}"^^xsd:dateTime ;',
        f'    dcterms:source "{source_file}" ;',
        f'    owl:versionInfo "{header_values["version"]}" ;',
        f"    owl:versionIRI <{VERSION_IRI}> ;",
        f"    owl:priorVersion <{PRIOR_VERSION_IRI}> ;",
        "    rdfs:seeAlso <https://sandraschaftner.github.io/orkg-properties-ontology-consolidation/> .",
        "",
    ]
    lines = turtle.split("\n")
    last_prefix = max((i for i, line in enumerate(lines) if line.strip().startswith("@prefix")), default=0)
    ttl_path = run.output_dir / f"opo-consolidated-{ONTOLOGY_VERSION}_{stamp}.ttl"
    ttl_path.write_text("\n".join(lines[: last_prefix + 1] + header + lines[last_prefix + 1 :]), encoding="utf-8")

    print(f"  {jsonld_path}")
    print(f"  {ttl_path}")
    return jsonld_path, ttl_path


def date_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# --------------------------------------------------------------------------------------
# Run context and entry point
# --------------------------------------------------------------------------------------


class RunContext:
    """Everything a run needs to know about where things live."""

    def __init__(self, run_dir: Path, workers: int, embedding_model: str, output_dir: Path):
        self.dir = run_dir
        self.workers = workers
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        (self.dir / "cache").mkdir(parents=True, exist_ok=True)

    def cache(self, name: str) -> DecisionCache:
        return DecisionCache(self.dir / "cache" / f"{name}.jsonl")


def load_input(path: Path, limit: Optional[int]) -> pd.DataFrame:
    """Load the property dump produced by fetch_properties.py."""
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise SystemExit(f"{path} is not a property dump (expected a list of properties)")
    df = pd.DataFrame(records).fillna("")
    df = df.drop_duplicates(subset=["uri", "label"])
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} properties from {path.name}")
    return df


def latest_input(directory: Path) -> Path:
    """Most recent property dump in opo/input/ (the .meta.json sidecars are not dumps)."""
    candidates = sorted(
        path for path in directory.glob("orkg_properties_*.json") if not path.name.endswith(".meta.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No property dump in {directory}. Run: python opo/fetch_properties.py"
        )
    return candidates[-1]


STEP_NAMES = {1: "step_1_lexical", 2: "step_2_quality", 3: "step_3_clustered", 4: "step_4_final", 5: "step_5_normalized"}


def main() -> None:
    parser = argparse.ArgumentParser(description="OPO consolidation pipeline (server edition)")
    parser.add_argument("--input", type=Path, help="property dump (default: newest in opo/input/)")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="where checkpoints and caches go")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "ontology", help="where the ontology goes")
    parser.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--end-step", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--limit", type=int, help="only process the first N properties (smoke test)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel LLM requests")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_API_URL))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("KISTE_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("KISTE_API_KEY is not set — put it in .env or export it")

    # Step 4 waits for keyboard input. Started without a terminal (nohup, a scheduler,
    # a notebook), it would hang silently instead of failing, so say so up front.
    if args.start_step <= 4 <= args.end_step and not sys.stdin.isatty():
        raise SystemExit(
            "Step 4 is the interactive expert review and needs a terminal.\n"
            "Run the unattended part with --end-step 3, then do step 4 interactively:\n"
            "  python opo/opo_consolidation.py --start-step 4 --end-step 4"
        )

    run = RunContext(args.run_dir, args.workers, args.embedding_model, args.output_dir)
    llm = LLMClient(args.base_url, api_key, args.model)
    llm.check()
    prompts = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8"))

    print(f"\nRun directory: {run.dir}")
    print(f"Steps {args.start_step}-{args.end_step}, {args.workers} parallel requests\n")

    mappings: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"run": {"started": datetime.now().isoformat(), "model": args.model, "limit": args.limit}}

    if args.start_step == 1:
        input_path = args.input or latest_input(HERE / "input")
        df = load_input(input_path, args.limit)
        stats["run"]["input_file"] = str(input_path)
    else:
        previous = STEP_NAMES[args.start_step - 1]
        print(f"Resuming from checkpoint '{previous}'")
        df, mappings, stats = load_checkpoint(run.dir, previous)

    steps = {
        1: (step_1_lexical, "step_1_lexical"),
        2: (step_2_quality, "step_2_quality"),
        3: (step_3_semantic, "step_3_clustered"),
        4: (step_4_interactive, "step_4_final"),
        5: (step_5_normalization, "step_5_normalized"),
    }

    for number in range(args.start_step, args.end_step + 1):
        function, checkpoint_name = steps[number]
        df = function(df, llm, prompts, mappings, stats, run)
        stats["run"]["llm_failures"] = llm.failures
        save_checkpoint(run.dir, checkpoint_name, df, mappings, stats)

    if args.end_step == 5:
        export_ontology(df, mappings, run, stats)

    stats["run"]["finished"] = datetime.now().isoformat()
    print(f"\nDone. {len(df)} canonical properties, {len(mappings)} mappings, {llm.failures} failed LLM calls.")


if __name__ == "__main__":
    main()
