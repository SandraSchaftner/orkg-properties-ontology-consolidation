# ORKG Properties Ontology Consolidated: LLM-Driven Refinement of Crowdsourced Knowledge

*The *OPO Consolidation Pipeline** is a hybrid neuro-symbolic framework designed to resolve semantic and lexical heterogeneity within the Open Research Knowledge Graph (ORKG) properties ontology.

This repository implements a complete consolidation pipeline that:

1.  **Deduplicates** properties lexically to resolve casing and spacing variations.
2.  **Filters** low-quality labels (non-predicates, objects, noise) using LLM-based verification.
3.  **Clusters** semantic synonyms using a hybrid approach combining **Qwen3-Embedding-8B** vectors and **GLM-4.5-Air** verification.
4.  **Validates** cluster decisions via a human-in-the-loop interactive review process.
5.  **Normalizes** labels to ORKG Best Practices and exports a backward-compatible ontology.

## Features

* **Backward Compatibility**: Generates `skos:exactMatch` and `owl:equivalentProperty` mappings to ensure no historical data is lost during consolidation.
* **Automated Evaluation**: Includes a rigorous evaluation suite comparing the consolidated ontology against a Gold Standard dataset (Vladyslav Nechakhin, Jennifer D’Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n)
* **Standardized Export**: Produces ontology artifacts in **JSON-LD** and **Turtle (.ttl)** formats.

## Prerequisites

* **Python 3.9+**
* **Hardware**: GPU (CUDA) or Apple Silicon (MPS) **highly recommended**.
    * *Note*: While the pipeline supports CPU execution, generating embeddings with the 8B parameter model (`Qwen3-Embedding-8B`) on a CPU will be significantly slower.
* **LLM Endpoint (OpenAI-Compatible)**:
    * The pipeline is designed to work with any **OpenAI-compatible API endpoint**. This can be a local inference server (e.g., LM Studio, vLLM, Ollama) or a remote provider.
    * **Default Configuration**: `http://127.0.0.1:1234/v1` (Configurable in `opo_consolidation.py`).

### Model Recommendations

The prompts and thresholds in this pipeline have been calibrated using the following models:

* **LLM**: `GLM-4.5-Air` 
* **Embeddings**: `Qwen/Qwen3-Embedding-8B` (Required for reproducing the clustering thresholds).


## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/SandraSchaftner/ORKG-Properties-Ontology-Consolidated.git](https://github.com/SandraSchaftner/ORKG-Properties-Ontology-Consolidated.git)
    cd ORKG-Properties-Ontology-Consolidated
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run Consolidation Pipeline

The main logic is contained in `opo_consolidation.py`. This script executes the 5-step consolidation process, saving checkpoints at every stage to allow for pausing and resuming.

```bash
python opo_consolidation.py
```
**Workflow Steps:**

1.  **Lexical Deduplication**: Merges exact string matches (case-insensitive) to reduce initial redundancy.
2.  **Quality Control**: Filters valid predicates using LLM verification, rejecting objects, topics, and noise.
3.  **Hybrid Clustering**:
    * Generates embeddings for all properties using `Qwen3-Embedding-8B`.
    * Performs loose agglomerative clustering to capture broad synonyms.
    * Uses the LLM to verify and refine clusters (Keep/Break decisions) to ensure precision.
4.  **Interactive Review**: Allows expert users to inspect and split complex clusters via CLI (Human-in-the-loop).
5.  **Normalization**: Standardizes labels (e.g., lowercase, singular) and generates final export files.

**Output:**

* `opo-consolidated_YYYY-MM-DD.jsonld` (JSON-LD Ontology)
* `opo-consolidated_YYYY-MM-DD.ttl` (Turtle Ontology)
* `checkpoints/` (contains detailed logs, intermediate CSVs, and mappings for every step)

### 2. Run Evaluation

The `Gold_Standard_evaluation.py` script assesses the quality of the consolidation using the *ORKG Gold Standard Property Dataset*.

```bash
python Gold_Standard_evaluation.py
```
**Metrics Calculated:**

* **Ambiguity Reduction**: Comparison of search candidate counts between the Original and Consolidated ontology to measure usability improvements.
* **Provenance Path Validation**: Verification that 100% of legacy properties map to a valid canonical URI via `skos:exactMatch`.
* **Statistical Significance**: Wilcoxon Signed-Rank Test results.

**Output:**

* `evaluation_results.json`: Detailed statistical report and mapping details.
* `figure_eval.png`: Violin plot visualizing the reduction in semantic ambiguity.

## Acknowledgment

The authors gratefully acknowledge Nechakhin, D'Souza, and Eger for publishing the ORKG Gold Standard Properties Dataset (Vladyslav Nechakhin, Jennifer D’Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n), which served as the evaluation dataset for this study.

## License

This project is licensed under **CC BY-SA 4.0**.