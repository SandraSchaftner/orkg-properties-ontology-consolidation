"""
OPO-Consolidated Evaluation

This script implements the evaluation protocol for the ontology OPO-Consolidated
as described in Schaftner & Gaedke (2026). It assesses the quality of the
consolidation using two metrics:

1.  Ambiguity Analysis (Usability): Measures the reduction in the number of search candidates returned for a
    given property label. Uses vector similarity to simulate semantic search.


2.  Provenance Path Validation (Backward Compatibility):
    Verifies that every legacy property in the Gold Standard dataset can be
    traced to a valid canonical property in the new ontology via
    'skos:exactMatch' mappings.

Input:
    - Gold Standard Property Dataset (Vladyslav Nechakhin, Jennifer D’Souza (2024). ORKG Properties and LLM-Generated Research Dimensions Evaluation Dataset [Data set]. LUIS. https://doi.org/10.25835/6oyn9d1n)
    - Original ORKG Property Dump
    - OPO Consolidated Ontology (JSON-LD)

Output:
    - `evaluation_results.json`: Quantitative metrics and mapping details.
    - `figure_eval.png`: Violin plot comparing search ambiguity (Original vs. Consolidated).
"""

import pandas as pd
import json
import ast
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Any, Tuple

GOLD_STANDARD_FILE = 'orkg_properties_llm_dimensions_dataset(1).csv'
ORIGINAL_PROPERTIES_FILE = 'orkg_properties_original_2025-12-31.json'
CONSOLIDATED_FILE = 'opo-consolidated_2026-01-07.jsonld'

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
ONTOLOGY_URI = "https://w3id.org/orkg-properties-ontology-consolidated"
THRESHOLD = 0.1


def load_gs(path: str) -> List[str]:
    """
    Loads unique property labels from the Gold Standard CSV.

    Args:
        path: File path to the Gold Standard dataset.

    Returns:
        A sorted list of unique property labels.
    """
    df = pd.read_csv(path, delimiter=';')
    all_props = []
    for x in df['orkg_properties']:
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list): all_props.extend(parsed)
        except:
            pass
    return sorted(list(set(all_props)))


def load_orig(path: str) -> List[Dict[str, str]]:
    """
    Loads the original (noisy) ORKG property set.

    Args:
        path: File path to the JSON export of the original ontology.

    Returns:
        A list of dictionaries containing 'uri' and 'label'.
    """
    print("Loading Original Data...")
    with open(path, 'r') as f:
        data = json.load(f)
    entries = []
    for item in data:
        if 'id' in item and 'label' in item:
            uri = f"https://orkg.org/property/{item['id']}"
            entries.append({'uri': uri, 'label': str(item['label'])})
    return entries


def load_cons_and_mappings(path: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Loads the consolidated ontology and mapping table.

    Args:
        path: File path to the OPO Consolidated JSON-LD file.

    Returns:
        Tuple containing:
        1. List of canonical properties.
        2. Dictionary mapping old URIs to new canonical URIs.
    """
    print("Loading Consolidated Ontology & Mappings...")
    with open(path, 'r') as f:
        data = json.load(f)
    canonicals = []
    mappings = {}

    for item in data.get('@graph', []):
        uri = item.get('@id')
        if not uri: continue

        # Extract Mappings (Backward Compatibility)
        if 'skos:exactMatch' in item:
            target = item['skos:exactMatch']
            if isinstance(target, dict): target = target.get('@id')
            mappings[uri] = target

        # Extract Canonical Properties
        defined = item.get('isDefinedBy')
        if isinstance(defined, dict): defined = defined.get('@id')
        if defined == ONTOLOGY_URI:
            lbl = item.get('label') or item.get('rdfs:label')
            if lbl: canonicals.append({'uri': uri, 'label': str(lbl)})

    return canonicals, mappings


def get_embeddings(texts: List[str], model: AutoModel, tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Generates dense vector embeddings for a list of texts using the Transformer model.

    Args:
        texts: List of strings to embed.
        model: Loaded Hugging Face model.
        tokenizer: Loaded Hugging Face tokenizer.

    Returns:
        A torch Tensor of shape (n_texts, hidden_size).
    """
    model.eval()
    device = model.device
    res = []
    for i in tqdm(range(0, len(texts), 32), desc="Generating Embeddings"):
        batch = texts[i:i + 32]
        inp = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp)
            mask = inp['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_emb = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            emb = F.normalize(sum_emb / sum_mask, p=2, dim=1)
            res.append(emb.cpu())
    return torch.cat(res, dim=0)


def plot_violin_final(orig_counts: List[int], cons_counts: List[int], threshold: float, sig_stars: str = "") -> None:
    """
    Generates the comparison Violin Plot (matches Figure 4 in the paper).

    Args:
        orig_counts: List of search result counts for the Original ontology.
        cons_counts: List of search result counts for the Consolidated ontology.
        threshold: The similarity threshold used.
        sig_stars: Significance notation (e.g., '***') to display on the plot.
    """
    print("\nGenerating Visualization (Violin Plot)...")

    # Filter zeros for visualization shape (consistent with paper method)
    orig_viz = [x for x in orig_counts if x > 0]
    cons_viz = [x for x in cons_counts if x > 0]
    if not orig_viz: orig_viz = [0]
    if not cons_viz: cons_viz = [0]

    data = [orig_viz, cons_viz]
    labels = ['Original ORKG', 'OPO-Consolidated']

    fig, ax = plt.subplots(figsize=(8, 6))

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    colors = ['#B0B0B0', '#494949']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    ax.boxplot(data, positions=[1, 2], widths=0.1,
               patch_artist=True,
               boxprops=dict(facecolor='white', alpha=0.9),
               medianprops=dict(color='black', linewidth=1.5),
               showfliers=True)

    # Stats Labels
    orig_avg = np.mean(orig_counts)
    cons_avg = np.mean(cons_counts)

    # Text Placement
    max_orig = max(orig_viz)
    max_cons = max(cons_viz)

    ax.text(1.1, max_orig * 0.95, f'Avg: {orig_avg:.1f}', va='top', fontsize=11, color='#555555')
    ax.text(2.1, max_cons * 0.95, f'Avg: {cons_avg:.1f}{sig_stars}', va='top', fontsize=11, color='black',
            fontweight='bold')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Number of Search Results (Similarity > {1 - threshold:.1f})', fontsize=11)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figure_eval.png', dpi=300)
    print("Figure saved as 'figure_eval.png'.")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Evaluation utilizing device: {device}")

    # 1. Load Data
    gs = load_gs(GOLD_STANDARD_FILE)
    orig_entries = load_orig(ORIGINAL_PROPERTIES_FILE)
    cons_entries, uri_mappings = load_cons_and_mappings(CONSOLIDATED_FILE)

    orig_uris = [e['uri'] for e in orig_entries]
    orig_labels = [e['label'] for e in orig_entries]
    cons_uris = [e['uri'] for e in cons_entries]
    cons_labels = [e['label'] for e in cons_entries]

    # 2. Generate Embeddings
    print("Initializing Model and Generating Embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)

    gs_vec = get_embeddings(gs, model, tokenizer)
    orig_vec = get_embeddings(orig_labels, model, tokenizer)
    cons_vec = get_embeddings(cons_labels, model, tokenizer)

    orig_counts = []
    cons_counts = []
    ambiguity_details = []

    path_details = []
    mapped_count = 0

    orig_gpu = orig_vec.to(device)
    cons_gpu = cons_vec.to(device)

    # 3. Calculate Ambiguity & Mapping Coverage
    print("Calculating Ambiguity & Mapping Statistics...")
    for i in range(len(gs)):
        q = gs_vec[i].unsqueeze(0).to(device)

        # A) Original Matches
        sim_o = torch.mm(q, orig_gpu.t())

        # Ambiguity Count
        count_o = ((1 - sim_o) < THRESHOLD).sum().item()
        orig_counts.append(count_o)

        # Path Validation: Find best Original match to check mapping validity
        best_o_idx = torch.argmax(sim_o).item()
        best_o_uri = orig_uris[best_o_idx]
        best_o_lbl = orig_labels[best_o_idx]

        # Check if this best match has a mapping to the new ontology
        mapped_target = uri_mappings.get(best_o_uri)
        status = "UNMAPPED"

        # Fallback: Is it already canonical (Self-Mapping)?
        if not mapped_target and best_o_uri in cons_uris:
            mapped_target = best_o_uri

        if mapped_target:
            status = "MAPPED"
            mapped_count += 1

        path_details.append({
            "query": gs[i],
            "original_match_label": best_o_lbl,
            "original_match_uri": best_o_uri,
            "status": status,
            "mapped_to_consolidated_uri": mapped_target
        })

        # B) Consolidated Matches (Ambiguity only)
        sim_c = torch.mm(q, cons_gpu.t())
        count_c = ((1 - sim_c) < THRESHOLD).sum().item()
        cons_counts.append(count_c)

        ambiguity_details.append({
            "query": gs[i],
            "hits_original": count_o,
            "hits_consolidated": count_c
        })

    # 4. Statistical Significance Test (Wilcoxon Signed-Rank Test)
    # Using Wilcoxon as data is paired and non-normally distributed
    stat, p_value = stats.wilcoxon(orig_counts, cons_counts)
    print(f"\nWilcoxon Signed-Rank Test: Statistic={stat}, p-value={p_value:.5e}")

    sig_stars = " (ns)"
    if p_value < 0.05: sig_stars = "*"
    if p_value < 0.01: sig_stars = "**"
    if p_value < 0.001: sig_stars = "***"

    print(f"Significance Level: {sig_stars}")
    print(f"Mapping Coverage: {mapped_count}/{len(gs)} ({mapped_count / len(gs) * 100:.1f}%)")

    # 5. Extract Extremes for Analysis
    top_orig = sorted(ambiguity_details, key=lambda x: x['hits_original'], reverse=True)[:10]
    top_cons = sorted(ambiguity_details, key=lambda x: x['hits_consolidated'], reverse=True)[:10]

    # 6. Export Results
    json_output = {
        "parameters": {"threshold": THRESHOLD, "model": MODEL_ID},
        "statistics": {
            "original_avg": float(np.mean(orig_counts)),
            "consolidated_avg": float(np.mean(cons_counts)),
            "wilcoxon_statistic": float(stat),
            "p_value": float(p_value),
        },
        "path_validation": {
            "total_queries": len(gs),
            "successful_mappings": mapped_count,
            "coverage_percentage": (mapped_count / len(gs)) * 100,
            "details": path_details
        },
        "top_ambiguous_examples": {
            "original_top_10": top_orig,
            "consolidated_top_10": top_cons
        }
    }

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)
    print("Evaluation results saved to 'evaluation_results.json'.")

    # 7. Generate Plot
    plot_violin_final(orig_counts, cons_counts, THRESHOLD, sig_stars)
    print("\nEvaluation Complete.")