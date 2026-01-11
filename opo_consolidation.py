"""
OPO Consolidation Pipeline

This script implements the consolidation framework for the Open Research Knowledge Graph (ORKG)
properties ontology, as presented in "ORKG Properties Ontology Consolidated: LLM-Driven
Refinement of Crowdsourced Knowledge for Machine-Actionability" (Schaftner & Gaedke, 2026).

The pipeline executes a five-step process to transform a noisy, crowdsourced property set
into a canonical ontology while ensuring backward compatibility:

1.  Lexical Deduplication: Resolves case variations and exact string matches.
2.  Semantic Quality Filtering: Filters non-predicate labels (objects, topics, noise)
    using LLM-based verification.
3.  Hybrid Semantic Clustering: A two-phase clustering approach using
    Vector Embeddings (Qwen3-Embedding-8B) and LLM Verification (GLM-4.5-Air).
4.  Interactive Expert Review: A human-in-the-loop stage to refine complex clusters.
5.  Normalization & Canonical Selection: Standardizes labels to ORKG Best Practices
    and generates the final ontology artifacts (JSON-LD, Turtle).

Dependencies:
    - sentence_transformers
    - scikit-learn
    - openai (compatible with local inference via LM Studio)
    - rdflib
    - pandas
"""

import os
import json
import yaml
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef


START_STEP = 1

INPUT_FILE = "orkg_properties_original_2025-12-31.json"
PROMPTS_FILE = "prompts.yaml"
CHECKPOINT_DIR = "checkpoints"

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL_NAME = "glm-4.5-air"
API_URL = "http://127.0.0.1:1234/v1"
API_KEY = "key"


THRESHOLD_LOOSE = 0.25
THRESHOLD_STRICT = 0.15

GARBAGE_LENGTH_THRESHOLD = 3
LINK_URI = "https://orkg.org/property/P41267"


def wait_for_user(action_needed: str) -> None:
    """
    Pauses execution to allow the user to manage local resources (e.g., swapping models).

    Args:
        action_needed: A description of the action the user needs to perform.
    """
    print("\n" + "-"*60)
    print(f"ACTION REQUIRED: {action_needed}")
    print("-"*60)
    input(">>> Press ENTER when ready to proceed... <<<")
    print("Resuming process...\n")

def get_openai_client() -> OpenAI:
    """Initializes and returns the OpenAI client configured for the local endpoint."""
    return OpenAI(base_url=API_URL, api_key=API_KEY, timeout=600.0, max_retries=2)

def strip_thinking(text: str) -> str:
    """
    Removes the 'chain of thought' or internal reasoning traces from LLM responses
    if strictly structured JSON is required.

    Args:
        text: The raw string response from the LLM.

    Returns:
        The cleaned string containing only the final JSON payload.
    """
    return re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def query_llm_schema(
    client: OpenAI,
    system_p: str,
    user_p: str,
    schema_def: Dict[str, Any],
    schema_name: str
) -> Dict[str, Any]:
    """
    Executes a structured output query against the LLM.

    Args:
        client: The OpenAI client instance.
        system_p: The system prompt defining the persona and rules.
        user_p: The user prompt containing the specific data to process.
        schema_def: The JSON schema definition for the expected output.
        schema_name: A unique name for the schema.

    Returns:
        A dictionary containing the parsed JSON response. Returns an empty dict on failure.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_p}, {"role": "user", "content": user_p}],
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema_def, "strict": True}}
        )
        content = strip_thinking(response.choices[0].message.content)
        return json.loads(content)
    except Exception as e:
        print(f"LLM Processing Error: {e}")
        return {}

def add_mapping(
    mappings_list: List[Dict[str, Any]],
    old_row: Union[pd.Series, Dict[str, Any]],
    new_uri: str,
    relation: str = "skos:exactMatch"
) -> None:
    """
    Records a semantic mapping between an original property and its canonical replacement.

    Args:
        mappings_list: The global list of mapping records to append to.
        old_row: The data row of the deprecated property.
        new_uri: The URI of the selected canonical property.
        relation: The SKOS or OWL predicate describing the relationship.
    """
    if old_row['uri'] == new_uri:
        return

    mappings_list.append({
        "original_uri": old_row['uri'],
        "original_id": old_row.get('id', 'Unknown'),
        "mapped_to_uri": new_uri,
        "relation": relation,
        "original_label": old_row.get('label', ''),
        "created_at": old_row.get('created_at', ''),
        "description": old_row.get('description', '')
    })

def generate_cluster_report(df: pd.DataFrame, filename: str) -> None:
    """
    Generates a text-based report of the current clusters for inspection.

    Args:
        df: The DataFrame containing property data and 'cluster_id'.
        filename: The path where the report will be saved.
    """
    print(f"Generating Cluster Report: {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"CLUSTER REPORT - Generated at {datetime.now()}\n")
        f.write("="*60 + "\n\n")
        if 'cluster_id' in df.columns:
            cluster_counts = df['cluster_id'].value_counts()
            for cid in cluster_counts.index:
                subset = df[df['cluster_id'] == cid]
                labels = sorted(subset['label'].unique().tolist())
                f.write(f"=== CLUSTER {cid} (Size: {len(subset)}) ===\n")
                for lbl in labels: f.write(f"- {lbl}\n")
                f.write("\n")
    print("Report saved successfully.")

def save_checkpoint(
    step_name: str,
    df: pd.DataFrame,
    mappings: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> None:
    """
    Persists the current pipeline state (Data, Mappings, Stats) to disk.

    Args:
        step_name: The identifier for the current pipeline step.
        df: The current DataFrame of valid properties.
        mappings: The cumulative list of backward-compatibility mappings.
        stats: Dictionary containing operational metrics.
    """
    folder = os.path.join(CHECKPOINT_DIR, step_name)
    if not os.path.exists(folder): os.makedirs(folder)
    print(f"\nCreating Checkpoint for '{step_name}' in {folder}...")

    df.to_csv(os.path.join(folder, "data.csv"), index=False)

    def convert(o):
        return int(o) if isinstance(o, np.int64) else o

    with open(os.path.join(folder, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, default=convert)

    with open(os.path.join(folder, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4, default=convert)

    with open(os.path.join(folder, "mappings.ttl"), "w", encoding="utf-8") as f:
        f.write("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n")
        f.write("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n")
        f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n")

        for m in mappings:
            safe_label = json.dumps(str(m['original_label']))
            f.write(f"<{m['original_uri']}> skos:exactMatch <{m['mapped_to_uri']}> ;\n")
            f.write(f"    owl:equivalentProperty <{m['mapped_to_uri']}> ;\n")
            f.write(f"    rdfs:label {safe_label} .\n")

    if 'cluster_id' in df.columns:
        generate_cluster_report(df, os.path.join(folder, "cluster_overview.txt"))

    print(f"Checkpoint '{step_name}' saved.")

def load_checkpoint_metadata(step_folder_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Restores Mappings and Statistics from a previous checkpoint.

    Args:
        step_folder_name: The name of the folder to load from.

    Returns:
        A tuple containing the mappings list and the stats dictionary.
    """
    mappings_path = os.path.join(CHECKPOINT_DIR, step_folder_name, "mappings.json")
    stats_path = os.path.join(CHECKPOINT_DIR, step_folder_name, "stats.json")

    mappings = []
    stats = {}

    if os.path.exists(mappings_path):
        with open(mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
        print(f"   -> RESTORED STATE: Loaded {len(mappings)} mappings from {step_folder_name}")
    else:
        print(f"   WARNING: No mappings.json found in {step_folder_name}. Provenance might be lost.")

    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    return mappings, stats

def step_1_lexical(
    df: pd.DataFrame,
    client: OpenAI,
    prompts: Dict[str, Any],
    mappings: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Step 1: Dedupes properties based on strict lexical matching (case-insensitive).
    """
    wait_for_user("Please start the LM Studio Server.")
    print("\nSTEP 1: Lexical Deduplication")

    df['label_lower'] = df['label'].astype(str).str.lower().str.strip()
    grouped = df.groupby('label_lower')
    kept_rows = []
    stats["step_1"] = {"total": len(df), "removed": 0}
    schema = {"type": "object", "properties": {"selected_uri": {"type": "string"}}, "required": ["selected_uri"], "additionalProperties": False}

    for _, group in tqdm(grouped, desc="Lexical Processing"):
        candidates = group.to_dict('records')
        if len(candidates) == 1:
            kept_rows.append(candidates[0])
            continue

        valid_uris = {c['uri'] for c in candidates}
        winner_uri = candidates[0]['uri']

        # Use LLM to pick the best URI if multiple exist for the same label
        if len(valid_uris) > 1:
            sys_p = prompts['step_1_disambiguation']['system']
            cand_json = json.dumps([{"uri": c['uri'], "id": c['id'], "desc": c['description']} for c in candidates])
            user_p = prompts['step_1_disambiguation']['user_template'].format(label=candidates[0]['label_lower'], candidates_json=cand_json)
            res = query_llm_schema(client, sys_p, user_p, schema, "disambiguation_schema")
            if res.get('selected_uri') in valid_uris:
                winner_uri = res.get('selected_uri')
        else:
            winner_uri = list(valid_uris)[0]

        winner_kept = False
        for cand in candidates:
            if cand['uri'] == winner_uri and not winner_kept:
                kept_rows.append(cand)
                winner_kept = True
            else:
                if cand['uri'] != winner_uri:
                    add_mapping(mappings, cand, winner_uri)
                stats["step_1"]["removed"] += 1

    return pd.DataFrame(kept_rows).drop(columns=['label_lower'])

def step_2_quality(
    df: pd.DataFrame,
    client: OpenAI,
    prompts: Dict[str, Any],
    mappings: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Step 2: Filters out properties that are not valid predicates (e.g., Objects, Topics, Noise).
    """
    print("\nSTEP 2: Quality Control (VIP Protection + Semantic Check)")
    wait_for_user("Please ensure LM Studio is running (Quality Check).")

    accepted_rows = []
    stats["step_2"] = {"checked": len(df), "rejected_regex": 0, "rejected_llm": 0}
    schema = {"type": "object", "properties": {"is_valid": {"type": "boolean"}}, "required": ["is_valid"], "additionalProperties": False}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Quality Filtering"):
        label = str(row['label']).strip()
        uri = row['uri']
        desc = str(row.get('description', ''))

        if uri == LINK_URI:
            accepted_rows.append(row)
            continue

        # Regex Check (Filter out auto-generated IDs)
        if re.match(r'^(orkg:)?(P|R)\d+(\s.*)?$', label, re.IGNORECASE):
            add_mapping(mappings, row, LINK_URI, relation="skos:exactMatch")
            stats["step_2"]["rejected_regex"] += 1
            continue

        is_valid = True
        word_count = len(label.split())

        # Check A: Long labels (often sentences or descriptions)
        if word_count > GARBAGE_LENGTH_THRESHOLD:
            sys_p = prompts['step_2_long_label_check']['system']
            user_p = prompts['step_2_long_label_check']['user_template'].format(label=label, description=desc)
            if not query_llm_schema(client, sys_p, user_p, schema, "qc_long").get('is_valid', True):
                is_valid = False

        # Check B: Standard Semantic Check
        if is_valid:
            sys_p = prompts['step_2_semantic_check']['system']
            user_p = prompts['step_2_semantic_check']['user_template'].format(label=label, description=desc)
            if not query_llm_schema(client, sys_p, user_p, schema, "qc_sem").get('is_valid', True):
                is_valid = False

        if is_valid:
            accepted_rows.append(row)
        else:
            stats["step_2"]["rejected_llm"] += 1
            add_mapping(mappings, row, LINK_URI, relation="skos:exactMatch")

    return pd.DataFrame(accepted_rows)

def step_3_semantic(
    df: pd.DataFrame,
    client: OpenAI,
    prompts: Dict[str, Any],
    mappings: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Step 3: Performs Hybrid Semantic Clustering.

    Process:
    1.  Vector Clustering (Loose Threshold): Groups potentially similar terms.
    2.  LLM Validation: Checks if the cluster contains distinct concepts.
    3.  Re-Clustering (Strict Threshold): Splits clusters rejected by the LLM.
    """
    print(f"\nSTEP 3: HYBRID Semantic Clustering")
    print(f"Strategy: Loose Cluster ({THRESHOLD_LOOSE}) -> LLM Check -> Strict Re-Cluster ({THRESHOLD_STRICT})")

    wait_for_user("Please STOP LM Studio (to free VRAM for Embeddings).")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True, device=device)
    df = df.reset_index(drop=True)

    print("Calculating embeddings...")
    embeddings = model.encode(df['label'].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=True)

    print("Freeing VRAM...")
    del model
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nPhase A: Loose Clustering (Threshold {THRESHOLD_LOOSE})...")
    cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESHOLD_LOOSE, metric='cosine', linkage='average')
    loose_labels = cluster_model.fit_predict(embeddings)

    wait_for_user("Please START LM Studio (for Cluster Validation).")

    df['temp_cluster'] = loose_labels
    final_cluster_map = {}

    stats["step_3"] = {
        "loose_clusters": len(np.unique(loose_labels)),
        "llm_kept": 0,
        "llm_broken": 0,
        "decisions": []
    }
    schema = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["KEEP", "BREAK"]}}, "required": ["decision"], "additionalProperties": False}

    print("\nPhase B & C: LLM Validation & Strict Refinement...")

    cluster_sizes = df['temp_cluster'].value_counts()
    sorted_clusters = cluster_sizes.index.tolist()
    current_cluster_id_counter = 0

    for cid in tqdm(sorted_clusters, desc="Validating Clusters"):
        subset_indices = df[df['temp_cluster'] == cid].index.tolist()
        subset_labels = df.loc[subset_indices, 'label'].unique().tolist()
        log_entry = {
            "loose_cluster_id": int(cid),
            "labels": subset_labels,
            "decision": "UNKNOWN",
            "strict_subclusters": []
        }

        if len(subset_indices) == 1:
            final_cluster_map[subset_indices[0]] = current_cluster_id_counter
            current_cluster_id_counter += 1
            log_entry["decision"] = "SINGLETON"
            stats["step_3"]["decisions"].append(log_entry)
            continue

        sys_p = prompts['step_3_cluster_validation']['system']
        user_p = prompts['step_3_cluster_validation']['user_template'].format(labels=json.dumps(subset_labels))

        res = query_llm_schema(client, sys_p, user_p, schema, "cluster_val")
        decision = res.get('decision', 'BREAK')
        log_entry["decision"] = decision

        if decision == "KEEP":
            for idx in subset_indices:
                final_cluster_map[idx] = current_cluster_id_counter
            current_cluster_id_counter += 1
            stats["step_3"]["llm_kept"] += 1

        else:
            stats["step_3"]["llm_broken"] += 1

            subset_embeds = embeddings[subset_indices]
            strict_model = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESHOLD_STRICT, metric='cosine', linkage='average')
            strict_labels = strict_model.fit_predict(subset_embeds)

            unique_sub_ids = np.unique(strict_labels)
            for sub_id in unique_sub_ids:
                sub_mask = (strict_labels == sub_id)
                original_indices_in_sub = [subset_indices[i] for i in range(len(subset_indices)) if sub_mask[i]]

                sub_labels = df.loc[original_indices_in_sub, 'label'].tolist()
                log_entry["strict_subclusters"].append({
                    "sub_id": int(sub_id),
                    "labels": sub_labels
                })

                for idx in original_indices_in_sub:
                    final_cluster_map[idx] = current_cluster_id_counter
                current_cluster_id_counter += 1

        stats["step_3"]["decisions"].append(log_entry)

    print(f"\nStats: Kept {stats['step_3']['llm_kept']} clusters, Broke {stats['step_3']['llm_broken']} clusters.")

    final_ids = []
    for i in range(len(df)):
        final_ids.append(final_cluster_map[i])

    df['cluster_id'] = final_ids
    df = df.drop(columns=['temp_cluster'])
    del embeddings

    print(f"-> Final Result: {df['cluster_id'].nunique()} clusters.")
    return df

def step_4_interactive(
    df: pd.DataFrame,
    client: OpenAI,
    prompts: Dict[str, Any],
    mappings: List[Dict[str, Any]],
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Step 4: Interactive human review of clusters. Allows splitting of clusters via embeddings.
    """
    print("\nSTEP 4: Interactive Review & Merge")

    if "step_4" not in stats:
        stats["step_4"] = {"interactive_splits": [], "merged": 0}

    report_path = os.path.join(CHECKPOINT_DIR, "step_3_clustered", "cluster_overview.txt")
    print(f"TIP: Please open '{report_path}' to identify clusters requiring review.")
    wait_for_user("Please STOP LM Studio (Embedding model required for splitting).")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading embedding model (Step 4)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True, device=device)
    df = df.reset_index(drop=True)
    embeddings = model.encode(df['label'].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=True)

    while True:
        print("\n" + "=" * 40)
        top = df['cluster_id'].value_counts().head(5)
        print(f"Largest clusters currently: \n{top}")
        user_in = input("\nInspect Cluster ID? (Enter Number or 'no' to finish): ").strip()
        if user_in.lower() in ['no', 'n', 'exit']:
            break
        try:
            cid = int(user_in)
        except ValueError:
            continue

        subset = df[df['cluster_id'] == cid]
        print(f"\nCluster {cid}: {subset['label'].tolist()}")

        if input("Split this cluster? (y/n): ").lower() == 'y':
            centers_in = input("Enter new center terms separated by commas (e.g., 'is, adopt'): ").strip()
            if centers_in:
                split_log = {
                    "cluster_id": cid,
                    "centers_input": centers_in,
                    "original_labels": subset['label'].tolist(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                stats["step_4"]["interactive_splits"].append(split_log)
                print(f"Action logged.")

                centers = [c.strip() for c in centers_in.split(',')]
                c_embeds = model.encode(centers, normalize_embeddings=True).astype(np.float32)
                sub_indices = subset.index
                sub_embeds = embeddings[sub_indices].astype(np.float32)

                base_id = int(time.time())
                for i, idx in enumerate(sub_indices):
                    scores = np.dot(c_embeds, sub_embeds[i])
                    df.at[idx, 'cluster_id'] = base_id + np.argmax(scores)
                print("Cluster split completed.")

    del model
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    wait_for_user("Please START LM Studio (Merging Phase).")
    final_rows = []
    grouped = df.groupby('cluster_id')

    schema = {"type": "object", "properties": {"selected_uri": {"type": "string"}}, "required": ["selected_uri"], "additionalProperties": False}

    for cid, group in tqdm(grouped, desc="Merging Clusters"):
        candidates = group.to_dict('records')
        if len(candidates) == 1:
            final_rows.append(candidates[0])
            continue

        valid_uris = {c['uri'] for c in candidates}

        winner_uri = sorted(candidates, key=lambda x: len(str(x['label'])))[0]['uri']

        sys_p = prompts['step_4_cluster_selection']['system']
        cand_json = json.dumps([{"uri": c['uri'], "label": c['label']} for c in candidates])
        user_p = prompts['step_4_cluster_selection']['user_template'].format(candidates_json=cand_json)

        res = query_llm_schema(client, sys_p, user_p, schema, "merge")
        if res.get('selected_uri') in valid_uris:
            winner_uri = res.get('selected_uri')

        for cand in candidates:
            if cand['uri'] == winner_uri:
                final_rows.append(cand)
            else:
                add_mapping(mappings, cand, winner_uri)
                stats["step_4"]["merged"] += 1

    return pd.DataFrame(final_rows)


def step_5_normalization(
    df: pd.DataFrame,
    client: OpenAI,
    prompts: Dict[str, Any],
    stats: Dict[str, Any],
    mappings: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Step 5: Normalizes labels (e.g., lowercase, singular) and performs final deduplication.

    This step ensures that if multiple properties are normalized to the same string
    (e.g., "Methods" -> "method" and "Method" -> "method"), they are merged into a single URI.
    """
    print("\nSTEP 5: Label Normalization & Final Deduplication")
    wait_for_user("Please ensure LM Studio is running.")

    stats["step_5"] = {"normalized": 0, "final_duplicates_merged": 0}
    schema_norm = {"type": "object", "properties": {"clean_label": {"type": "string"}}, "required": ["clean_label"], "additionalProperties": False}
    schema_dedup = {"type": "object", "properties": {"selected_uri": {"type": "string"}}, "required": ["selected_uri"], "additionalProperties": False}

    df['clean_label_temp'] = df['label'].astype(str)

    print("Phase 1: Computing normalized labels...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Normalizing"):
        original_label = str(row['label'])

        # Optimization: Skip LLM for simple cases
        if original_label.islower() and " " not in original_label and len(original_label) > 2:
            clean_label = original_label
        else:
            sys_p = prompts['step_5_normalization']['system']
            user_p = prompts['step_5_normalization']['user_template'].format(label=original_label)
            res = query_llm_schema(client, sys_p, user_p, schema_norm, "norm")
            clean_label = res.get('clean_label', original_label)

        if clean_label != original_label:
            df.at[idx, 'clean_label_temp'] = clean_label
            stats["step_5"]["normalized"] += 1

    print("Phase 2: Resolving collisions (Final Deduplication)...")
    duplicates = df[df.duplicated(subset=['clean_label_temp'], keep=False)]
    rows_to_drop_indices = []

    if not duplicates.empty:
        print(f"Found {len(duplicates)} properties involved in collisions. Resolving...")
        grouped = df.groupby('clean_label_temp')

        for label, group in tqdm(grouped, desc="Deduplicating"):
            if len(group) == 1: continue

            candidates = group.to_dict('records')
            cand_json = json.dumps([
                {"uri": c['uri'], "id": c.get('id', ''), "desc": c.get('description', '')}
                for c in candidates
            ])

            sys_p = prompts['step_5_final_dedup']['system']
            user_p = prompts['step_5_final_dedup']['user_template'].format(label=label, candidates_json=cand_json)

            res = query_llm_schema(client, sys_p, user_p, schema_dedup, "final_dedup")
            winner_uri = res.get('selected_uri')

            # Fallback heuristic
            valid_uris = {c['uri'] for c in candidates}
            if winner_uri not in valid_uris:
                candidates.sort(key=lambda x: (len(str(x.get('id', ''))), str(x['uri'])))
                winner_uri = candidates[0]['uri']

            # Merge
            for cand in candidates:
                if cand['uri'] != winner_uri:
                    add_mapping(mappings, cand, winner_uri)
                    idx_to_drop = df.index[df['uri'] == cand['uri']].tolist()
                    rows_to_drop_indices.extend(idx_to_drop)
                    stats["step_5"]["final_duplicates_merged"] += 1

    df = df.drop(index=list(set(rows_to_drop_indices)))
    print("Phase 3: Finalizing...")
    df['label'] = df['clean_label_temp']
    df = df.drop(columns=['clean_label_temp'])

    print(f"-> Final property count: {len(df)}")
    return df

def main():
    print(f"--- OPO CONSOLIDATION PIPELINE (Start Step: {START_STEP}) ---")
    client = get_openai_client()
    with open(PROMPTS_FILE, "r") as f:
        prompts = yaml.safe_load(f)

    all_mappings = []
    all_stats = {}

    # 1. Lexical
    df_s1 = None
    if START_STEP == 1:
        with open(INPUT_FILE, 'r') as f: data = json.load(f)
        df_in = pd.DataFrame(data).fillna("")
        df_in = df_in.drop_duplicates(subset=['uri', 'label'])
        print(f"Loaded {len(df_in)} raw records.")

        df_s1 = step_1_lexical(df_in, client, prompts, all_mappings, all_stats)
        save_checkpoint("step_1", df_s1, all_mappings, all_stats)
    elif START_STEP > 1:
        print("Skipping Step 1 (Loading Checkpoint)...")
        df_s1 = pd.read_csv(os.path.join(CHECKPOINT_DIR, "step_1", "data.csv")).fillna("")
        all_mappings, all_stats = load_checkpoint_metadata("step_1")

    # 2. Quality
    df_s2 = None
    if START_STEP <= 2:
        df_s2 = step_2_quality(df_s1, client, prompts, all_mappings, all_stats)
        save_checkpoint("step_2_quality", df_s2, all_mappings, all_stats)
    elif START_STEP > 2:
        print("Skipping Step 2 (Loading Checkpoint)...")
        df_s2 = pd.read_csv(os.path.join(CHECKPOINT_DIR, "step_2_quality", "data.csv")).fillna("")
        all_mappings, all_stats = load_checkpoint_metadata("step_2_quality")

    # 3. Hybrid Clustering
    df_s3 = None
    if START_STEP <= 3:
        df_s3 = step_3_semantic(df_s2, client, prompts, all_mappings, all_stats)
        save_checkpoint("step_3_clustered", df_s3, all_mappings, all_stats)
    elif START_STEP > 3:
        print("Skipping Step 3 (Loading Checkpoint)...")
        df_s3 = pd.read_csv(os.path.join(CHECKPOINT_DIR, "step_3_clustered", "data.csv")).fillna("")
        all_mappings, all_stats = load_checkpoint_metadata("step_3_clustered")

    # 4. Interactive
    df_s4 = None
    if START_STEP <= 4:
        df_s4 = step_4_interactive(df_s3, client, prompts, all_mappings, all_stats)
        save_checkpoint("step_4_final", df_s4, all_mappings, all_stats)
    elif START_STEP > 4:
        print("Skipping Step 4 (Loading Checkpoint)...")
        df_s4 = pd.read_csv(os.path.join(CHECKPOINT_DIR, "step_4_final", "data.csv")).fillna("")
        all_mappings, all_stats = load_checkpoint_metadata("step_4_final")

    # 5. Normalization & Export
    if START_STEP <= 5:
        df_final = step_5_normalization(df_s4, client, prompts, all_stats, all_mappings)
        save_checkpoint("step_5_normalized", df_final, all_mappings, all_stats)

        print("\nGenerating Output Artifacts (JSON-LD & Turtle)...")
        export_cols = ['uri', 'id', 'label', 'description', 'created_at']
        final_cols = [c for c in export_cols if c in df_final.columns]
        records = df_final[final_cols].to_dict(orient='records')
        ts = datetime.now().strftime("%Y-%m-%d")

        ontology_uri = "https://w3id.org/orkg-properties-ontology-consolidated"
        version = "1.0.0"

        json_ld_graph = []

        json_ld_graph.append({
            "@id": ontology_uri,
            "@type": "owl:Ontology",
            "dcterms:title": "ORKG Properties Ontology Consolidated (OPO-Consolidated)",
            "dcterms:description": "A consolidated ontology of research properties derived from ORKG.",
            "dcterms:creator": "OPO Consolidation Pipeline",
            "dcterms:created": {"@value": ts, "@type": "xsd:date"},
            "owl:versionInfo": version
        })

        for row in records:
            json_ld_graph.append({
                "@id": row['uri'],
                "@type": "rdf:Property",
                "label": row.get('label', ''),
                "skos:prefLabel": row.get('label', ''),
                "description": row.get('description', ''),
                "id": row.get('id', ''),
                "created_at": {"@value": row.get('created_at', ''), "@type": "xsd:dateTime"},
                "isDefinedBy": ontology_uri
            })

        for m in all_mappings:
            json_ld_graph.append({
                "@id": m['original_uri'],
                "@type": "rdf:Property",
                "skos:exactMatch": {"@id": m['mapped_to_uri']},
                "owl:equivalentProperty": {"@id": m['mapped_to_uri']},
                "rdfs:label": m.get('original_label', '')
            })

        json_ld = {
            "@context": {
                "orkg": "https://orkg.org/property/",
                "opo": f"{ontology_uri}#",
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
                "created_at": "dcterms:created"
            },
            "@graph": json_ld_graph
        }

        with open(f"opo-consolidated_{ts}.jsonld", "w", encoding="utf-8") as f:
            json.dump(json_ld, f, indent=2)


        print("Generating Turtle serialization...")

        g = Graph()
        g.parse(data=json.dumps(json_ld), format='json-ld')

        ORKG = Namespace("https://orkg.org/property/")
        OPO = Namespace(f"{ontology_uri}#")
        DCTERMS = Namespace("http://purl.org/dc/terms/")
        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

        g.bind("orkg", ORKG)
        g.bind("opo", OPO)
        g.bind("dcterms", DCTERMS)
        g.bind("skos", SKOS)
        g.bind("owl", OWL)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

        ont_subj = URIRef(ontology_uri)

        # Extract metadata for custom header
        title_lit = g.value(ont_subj, DCTERMS.title)
        desc_lit = g.value(ont_subj, DCTERMS.description)
        creator_lit = g.value(ont_subj, DCTERMS.creator)
        created_lit = g.value(ont_subj, DCTERMS.created)
        version_lit = g.value(ont_subj, OWL.versionInfo)

        g.remove((ont_subj, None, None))
        ttl_data = g.serialize(format='turtle')

        header_block = [
            "",
            f"<{ontology_uri}> a owl:Ontology ;",
            f'    dcterms:title "{title_lit}"@en ;',
            f'    dcterms:description "{desc_lit}"@en ;',
            f'    dcterms:creator "{creator_lit}" ;',
            f'    dcterms:created "{created_lit}"^^xsd:date ;',
            f'    owl:versionInfo "{version_lit}" ;',
            "rdfs:seeAlso <https://sandraschaftner.github.io/orkg-properties-ontology-consolidation/> .",
            ""
        ]

        lines = ttl_data.split('\n')
        last_prefix_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('@prefix'):
                last_prefix_index = i

        final_lines = lines[:last_prefix_index + 1] + header_block + lines[last_prefix_index + 1:]
        final_output = '\n'.join(final_lines)

        ttl_filename = f"opo-consolidated_{ts}.ttl"
        with open(ttl_filename, "w", encoding="utf-8") as f:
            f.write(final_output)

        print(f"DONE. Generated opo-consolidated_{ts}.jsonld and {ttl_filename}. Total Mappings: {len(all_mappings)}")


if __name__ == "__main__":
    main()