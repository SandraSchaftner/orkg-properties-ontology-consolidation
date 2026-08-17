# ORKG Properties Ontology Consolidated: LLM-Driven Refinement of Crowdsourced Knowledge

The **OPO Consolidation Pipeline** is a hybrid neuro-symbolic framework designed to resolve semantic and lexical heterogeneity within the Open Research Knowledge Graph (ORKG) properties ontology.

This repository implements a complete consolidation pipeline that:

1.  **Deduplicates** properties lexically to resolve casing and spacing variations.
2.  **Filters** low-quality labels (non-predicates, objects, noise) using LLM-based verification.
3.  **Clusters** semantic synonyms using a hybrid approach combining embedding vectors and LLM verification.
4.  **Validates** cluster decisions via a human-in-the-loop interactive review process.
5.  **Normalizes** labels to ORKG Best Practices and exports a backward-compatible ontology.

## Versions

> **The paper describes v1.0.0.** All results reported in Schaftner & Gaedke (2026) — including the
> evaluation numbers and the violin plot (Figure 4) — were produced from **v1.0.0**. That artifact
> is preserved unchanged in this repository and remains permanently citable under its own version
> IRI, so readers of the paper can always reach exactly what was evaluated.

**v1.1.0 is the current release.** It applies the same method to a newer ORKG snapshot with a
different chat model, and fixes one defect that affected the v1.0.0 result (see below).

| | v1.0.0 | v1.1.0 |
| --- | --- | --- |
| released | 2026-01-07 | 2026-08-17 |
| input snapshot | 2025-12-31, 13,009 properties | 2026-08-14, 14,193 properties |
| canonical properties | 2,865 | 3,175 |
| backward-compatibility mappings | 10,120 | 11,018 |
| chat model | `GLM-4.5-Air` (local, LM Studio) | `Qwen3.6-35B-A3B-MLX-8bit` (inference server) |
| embedding model | `Qwen/Qwen3-Embedding-8B` | `Qwen/Qwen3-Embedding-8B` (unchanged) |
| Turtle | [`ontology/opo-consolidated-1.0.0.ttl`](ontology/opo-consolidated-1.0.0.ttl) | [`ontology/opo-consolidated-1.1.0.ttl`](ontology/opo-consolidated-1.1.0.ttl) |
| JSON-LD | [`opo-consolidated_2026-01-07.jsonld`](opo-consolidated_2026-01-07.jsonld) | [`ontology/opo-consolidated-1.1.0.jsonld`](ontology/opo-consolidated-1.1.0.jsonld) |
| version IRI | `https://w3id.org/orkg-properties-ontology-consolidated/1.0.0` | `https://w3id.org/orkg-properties-ontology-consolidated/1.1.0` |
| pipeline code | `opo_consolidation.py` (repository root) | [`v1.1.0/opo/`](v1.1.0/opo/) |
| release notes | — | [`v1.1.0/opo/CHANGES.md`](v1.1.0/opo/CHANGES.md) |

Cite a **version IRI** when the exact artifact matters. The unversioned IRI
`https://w3id.org/orkg-properties-ontology-consolidated` tracks the current release and its target
changes between versions.

### What changed in v1.1.0

Three differences matter; [`v1.1.0/opo/CHANGES.md`](v1.1.0/opo/CHANGES.md) has the full list with
measurements.

1.  **A different chat model.** On a balanced sample of 200 properties from the v1.0.0 run, the two
    models agreed on 82.5% of decisions. Where they differ, the new model follows the written
    criteria of `prompts.yaml` more closely in most cases, and it is stricter overall (49% of
    properties rejected versus 40%). The embedding model is unchanged.
2.  **A newer input snapshot.** 14,193 instead of 13,009 properties. Notably `P4077` ("source
    code"), the single most used predicate in ORKG with 112k statements, was not covered by v1.0.0
    at all and is now.
3.  **The link property is kept out of the clustering.** `P41267` collects everything the quality
    filter rejects. Because it carries a label like any other property, it took part in the
    clustering itself and in v1.0.0 ended up merged into "links". Resolving a mapping chain to its
    end therefore returned a legitimate property for every rejected one. In v1.1.0 it is set aside
    before clustering and stays a terminal node. This is the only change that alters the *result*
    rather than the execution.

## Evaluation

Both versions were evaluated with the protocol described in the paper (ambiguity analysis and
provenance path validation) against the *ORKG Gold Standard Property Dataset*, using the same gold
standard, the same embedding model and the same threshold.

| metric | v1.0.0 | v1.1.0 |
| --- | --- | --- |
| avg. search candidates, original ORKG | 47.44 | 45.53 |
| avg. search candidates, consolidated | 7.98 | 10.16 |
| **ambiguity reduction** | **83.2%** | **77.7%** |
| Wilcoxon signed-rank, *p* | 9.58 × 10⁻⁴¹ | 4.12 × 10⁻⁴¹ |
| provenance path coverage | 253/253 = 100% | 253/253 = 100% |

Both metrics hold for v1.1.0: the reduction in semantic ambiguity is highly significant, and every
gold-standard property still resolves to a canonical property, so no historical data is lost.

The reduction is somewhat lower than in v1.0.0 for two measured reasons. First, v1.1.0 contains
10.8% more canonical properties, and its more conservative clustering deliberately keeps
near-synonyms apart. Second, v1.0.0's figure was flattered by a normalisation defect: 163 of its
canonical labels retained underscores (`evaluation_scope`), which placed them outside
natural-language space and made them invisible to the search metric. Fixing that in v1.1.0 accounts
for part of the difference. It is therefore not a regression in consolidation quality.

Results per version: `evaluation_results_<version>.json` and `figure_eval_<version>.png` (v1.0.0
uses the unsuffixed names `evaluation_results.json` and `figure_eval.png`).

## Documentation

Human-readable documentation of the ontology is published at
<https://sandraschaftner.github.io/orkg-properties-ontology-consolidation/>.

## Prerequisites

*   **Python 3.9+** (the v1.1.0 artifacts were produced and evaluated on Python 3.14).
*   **Hardware**: GPU (CUDA) or Apple Silicon (MPS) **highly recommended**.
    *   *Note*: While the pipeline supports CPU execution, generating embeddings with the 8B parameter model (`Qwen3-Embedding-8B`) on a CPU will be significantly slower.
*   **LLM Endpoint (OpenAI-Compatible)**: any OpenAI-compatible API endpoint, local (LM Studio, vLLM, Ollama) or remote. Needed for the consolidation pipeline only — **the evaluation script needs no LLM and no API key.**

### Model recommendations

The prompts and thresholds were calibrated with `GLM-4.5-Air` (v1.0.0) and re-run with
`Qwen3.6-35B-A3B-MLX-8bit` (v1.1.0). `Qwen/Qwen3-Embedding-8B` is required for reproducing the
clustering thresholds in both versions.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/SandraSchaftner/orkg-properties-ontology-consolidation.git
    cd orkg-properties-ontology-consolidation
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    Versions are pinned. The evaluation compares embedding similarities against a hard threshold and
    is sensitive to the numerical precision of the embedding model, so an unpinned environment can
    silently change the reported numbers.

## Usage

### 1. Run the consolidation pipeline

The v1.0.0 pipeline is `opo_consolidation.py` in the repository root; run it with
`python opo_consolidation.py`. The reworked v1.1.0 pipeline lives in
[`v1.1.0/opo/`](v1.1.0/opo/) and is driven by command-line flags — it runs unattended, resumes at
the level of individual model decisions, and fetches its input from the ORKG API. See
[`v1.1.0/opo/README.md`](v1.1.0/opo/README.md).

**Workflow steps:**

1.  **Lexical Deduplication**: Merges exact string matches (case-insensitive) to reduce initial redundancy.
2.  **Quality Control**: Filters valid predicates using LLM verification, rejecting objects, topics, and noise.
3.  **Hybrid Clustering**: Generates embeddings for all properties, performs loose agglomerative clustering to capture broad synonyms, then uses the LLM to verify and refine clusters (Keep/Break decisions).
4.  **Interactive Review**: Allows expert users to inspect, split and merge clusters via CLI (human-in-the-loop).
5.  **Normalization**: Standardizes labels (e.g. lowercase, singular) and generates the final export files in JSON-LD and Turtle.

### 2. Run the evaluation

```bash
python Gold_Standard_evaluation.py            # defaults to v1.1.0
```

Input and output paths are command-line arguments (`--original`, `--consolidated`, `--results`,
`--figure`); the header of the script gives the exact invocation that reproduces the published
v1.0.0 numbers. The original property dump must be the snapshot the ontology was built from,
otherwise the comparison mixes two different states of ORKG.

**Metrics calculated:**

*   **Ambiguity Reduction**: Comparison of search candidate counts between the original and the consolidated ontology to measure usability improvements.
*   **Provenance Path Validation**: Verification that 100% of legacy properties map to a valid canonical URI via `skos:exactMatch`.
*   **Statistical Significance**: Wilcoxon signed-rank test results.

## Features

*   **Backward Compatibility**: Generates `skos:exactMatch` and `owl:equivalentProperty` mappings to ensure no historical data is lost during consolidation.
*   **Automated Evaluation**: Includes a rigorous evaluation suite comparing the consolidated ontology against a Gold Standard dataset (Vladyslav Nechakhin, Jennifer D'Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n)
*   **Standardized Export**: Produces ontology artifacts in **JSON-LD** and **Turtle (.ttl)** formats.

## Acknowledgment

The authors gratefully acknowledge Nechakhin, D'Souza, and Eger for publishing the ORKG Gold Standard Properties Dataset (Vladyslav Nechakhin, Jennifer D'Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n), which served as the evaluation dataset for this study.

This work is supported by the European Union's HORIZON Research and Innovation Programme under grant agreement No 101120657, project ENFIELD (European Lighthouse to Manifest Trustworthy and Green AI).

## License

This project is licensed under **CC BY-SA 4.0**.
