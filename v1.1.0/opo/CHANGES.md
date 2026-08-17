# Release notes — OPO-Consolidated v1.1.0

Release notes for **v1.1.0** of the ontology behind *"ORKG Properties Ontology Consolidated:
LLM-Driven Refinement of Crowdsourced Knowledge for Machine-Actionability"* (Schaftner & Gaedke,
2026). The paper describes **v1.0.0**; this document records what changed on the way to v1.1.0,
relative to the original pipeline published with the paper (`opo_consolidation.py` in the
repository root).

This folder holds the reworked pipeline that produced the release. It was originally developed for
a separate property-usage analysis, which needed the ~14,000 ORKG predicates folded into canonical
concepts before counting how often properties are used per research field — otherwise the 23
different predicate ids that all mean "method" are counted as 23 separate properties. That is why
several sections below reason about downstream analysis; the artifacts themselves are a general
ontology release.

**The method is unchanged.** Same five steps, same prompts, same clustering thresholds, same
checkpoint layout, same output artifacts. Most of what changed is *how the pipeline runs*: where
the models live, how failures are survived, and how long it takes. Two changes affect the result
itself and are called out as such: the treatment of the link property (§10) and the normalisation
of underscore labels (§11).

## The release at a glance

14,193 properties in, 3,175 canonical properties and 11,018 mappings out — the two add up exactly
to the input. Every mapping chain resolves to a canonical property in at most two hops, with no
cycles and no dead ends, and 0 of roughly 35,000 LLM calls failed. Steps 1–3 ran 3.97 h unattended,
the manual expert review took 96 minutes (18 splits, 20 merges), step 5 a further 17 minutes.

The artifacts are `ontology/opo-consolidated-1.1.0.ttl` and `.jsonld`. They carry
`owl:versionIRI <https://w3id.org/orkg-properties-ontology-consolidated/1.1.0>` and
`owl:priorVersion <…/1.0.0>`, so the two releases are distinguishable as RDF rather than only by
file name.

---

## Overview

| Aspect | Original | Here |
| --- | --- | --- |
| Chat model | `glm-4.5-air` in LM Studio, locally | `Qwen3.6-35B-A3B-MLX-8bit` on the VSR inference server (Kiste) |
| Embedding model | `Qwen3-Embedding-8B`, local | unchanged, still local |
| Manual pauses | 5 × "start/stop LM Studio" | none |
| Requests | strictly sequential | 8 in parallel |
| Recovery after a crash | restart the whole step | resumes at the last decision |
| Input | static JSON export | fetched from the ORKG API |
| Parameters | edit constants in the file | command-line flags |
| HTTP client | `openai` package | `requests` |

---

## 1. Chat completions run on the inference server

The original ran the chat model in LM Studio on a MacBook. Because the 8B embedding model
and the chat model did not fit in memory at the same time, the script stopped five times and
asked the user to start or stop LM Studio by hand (`wait_for_user`). That made an unattended
run impossible.

Chat completions now go to `https://kiste.informatik.tu-chemnitz.de/v1`, an OpenAI-compatible
endpoint. The memory conflict disappears, and with it every manual pause. The pipeline can be
started and left alone.

The API key is read from the environment (`KISTE_API_KEY`), never from the source.

## 2. Embeddings stay local

The inference server offers chat completions but no embedding endpoint, so steps 3 and 4 keep
using `sentence-transformers` with `Qwen3-Embedding-8B` on the local machine (MPS on Apple
silicon, CUDA elsewhere). This is the only reason the heavy dependencies (`torch`,
`sentence-transformers`, `scikit-learn`) are still needed.

Embeddings are cached to disk per label set, so a repeated run does not reload the 14 GB model.

## 3. Parallel requests

The original sent one request at a time. All per-property decisions are independent and run at
`temperature=0`, so they can be issued concurrently without changing the result.

Measured against the server (single classification calls with the real prompts):

| parallel requests | throughput |
| --- | --- |
| 1 | 2.5 req/s |
| 4 | 3.6 req/s |
| 8 | 4.1 req/s |
| 16 | 4.2 req/s |

The server saturates at about 4 requests per second; beyond 8 workers only latency grows.
The default is therefore 8. Step 2 (the largest step, ~11,000 properties) takes roughly
2.5 hours, a full run four to five hours.

## 4. Resumable at decision level

The original wrote a checkpoint after each of the five steps. A crash three hours into step 2
meant three hours lost.

Every individual model decision is now appended to a JSONL cache under `runs/<name>/cache/`
the moment it is made, keyed by property URI (or by a hash of the cluster's labels). A restart
reads the cache and only asks about what is still unknown. Re-running a completed step costs
under a second and yields byte-identical results.

The between-step checkpoints are kept as they were, so `--start-step` still works the same way.

## 5. Input comes from the live API

`fetch_properties.py` pulls the current property set from `/api/predicates` and writes it in
exactly the field layout the original expected (`uri`, `id`, `label`, `description`,
`created_at`).

This keeps a release current: properties are added to ORKG over time but virtually never removed,
so a fresh dump is a superset of any earlier one. For a downstream consumer the recommended order
is to take the property dump immediately *after* downloading the data to be analysed, so the
consolidation covers every predicate that data can contain.

For reference, on 2026-08-14 the published ontology v1.0.0 covered 11,643 of the 14,207
predicates that existed at that point — 2,564 predicates (18%) had been created since the original
dump. Among them `P4077` ("source code"), the single most used predicate in ORKG with 112k
statements, which v1.0.0 did not cover at all.

## 6. Deliberately *not* changed: no request batching

Sending ten labels in one request is three times faster per label (0.29 s vs 0.90 s), because
the ~1,900-token system prompt is then paid once instead of ten times — the server does not
support prompt caching (`cached_tokens` stays 0).

It was measured on the same 200-property sample and rejected. Batching shifts the decision
boundary: the model keeps 114 of 200 properties in batches of ten versus 91 when asked one at
a time, and 25 of the 28 differences point the same way (reject → keep). The properties that
flip are exactly the domain-specific measurement variables and topics the prompt asks to
reject (`purgingGasFlow`, `Sleep efficiency (%)`, `display_density`, `Climate`, …).

Since a wrong "keep" leaves noise in the ontology and a wrong "reject" removes a property
silently, the speedup was not worth it. All calls are single-property calls.

## 7. `requests` instead of the `openai` package

The pipeline uses one endpoint with one request shape. Calling it directly removes a
dependency and makes the retry behaviour explicit (three attempts with exponential backoff,
no retry on 4xx). Structured outputs work exactly as before:
`response_format.json_schema` with `strict: true` is honoured by the server — verified, the
key names come back exactly as specified.

A useful side effect: with the schema enforced, a classification answer costs 6 output tokens.
Without it the model "thinks out loud" first and produces ~818 tokens for the same answer.
The schema is not just for parsing convenience, it is a 25× cost factor.

## 8. Configuration via command line

`START_STEP` and the file paths were constants to be edited in the source. They are flags now:

```
--input / --run-dir / --output-dir / --start-step / --end-step
--limit / --workers / --model / --base-url / --embedding-model
```

`--limit` in particular allows a smoke test over a few hundred properties before committing to
a full run.

## 9. Failures are counted, not silent

`query_llm_schema` returned `{}` on any error, and the callers then fell back to their default
(`is_valid=True`, `decision="BREAK"`). A network hiccup could therefore quietly change the
ontology. The behaviour is kept — a single failure should not abort a four-hour run — but every
failure is now printed and counted in `stats["run"]["llm_failures"]`, so a run with silent
degradation is recognisable afterwards.

Additionally the pipeline performs a health check against the server before starting, instead
of failing on the first of 11,000 calls.

## 10. The link property is kept out of the clustering

This is the one change that affects the *result* rather than only the execution.

Step 2 maps everything it rejects — objects, topics, noise — onto the generic link property
`P41267`, so that existing statements stay resolvable. `P41267` has a label like any other
property, so from step 3 onwards it was treated like any other property too: embedded,
clustered, and eligible to be merged into whichever cluster its label landed in.

That gives the collector a mapping of its own. In the v1.0.0 artifacts it ends up inside the
cluster around "links":

```
CSVW_Column  ->  P15175  ->  P41267  ->  P15325 ("links")
```

Resolving such a chain to its end therefore returns "links" for every rejected property, and the
distinction between "this was rejected" and "this is a link" is no longer visible to a consumer of
the ontology. For anything that counts property usage, that matters: the rejects are counted as a
legitimate and rather popular property.

`P41267` is therefore set aside before clustering, re-added with a cluster of its own, excluded
from label normalisation, and given precedence in any label collision it takes part in. It stays
a terminal node — which is what a collector for rejected properties needs to be, since its whole
purpose is to be recognisable as one.

**Note for consumers of v1.0.0:** the earlier release does *not* have this property, so chain
resolution against v1.0.0 should stop at `P41267` explicitly. Provenance validation is unaffected
in either version — in v1.1.0 `P41267` is itself a canonical property, so every legacy property
still resolves.

## 11. Normalisation no longer skips underscore labels

Step 5 skips the model for labels that are already a lowercase single word. `has_method`,
`benchmark_result` and `paper:venue` pass that test — they contain no space — so they were never
normalised, which is precisely backwards: those are the labels the normalisation exists for. The
condition now also treats `_` and `:` as word separators. In this run it changed 198 labels
(`evaluation_scope` → `evaluation scope`, `paper:links_source_information` → `links source
information`); genuine cases such as `male:female ratio` are kept by the model.

The published v1.0.0 shows the same pattern (163 of its 2,865 canonical labels contain an
underscore), so this affects any earlier artifact as well.

## 12. Step 4 is still interactive — and can now merge, not only split

The expert review remains manual, as in the original. Only the LM Studio pauses around it are
gone. Because the steps are checkpointed, you can run steps 1–3 unattended, do the review
whenever you have time (`--start-step 4`), and let step 5 finish afterwards.

The review itself gained three things, prompted by what the new run's clustering actually looks
like. Compared with the v1.0.0 run, the model breaks clusters far more often (31.6% of the
validated clusters versus 13.3%), so step 3 now produces finer groups: for the clusters that
were split by hand in v1.0.0, the new run yields 4.3 groups on average where the manual split
produced 2.7, and 2,011 of the 3,259 clusters hold a single label. Finer is the safer direction —
merging too much cannot be undone by inspection — but a concept spread over several clusters
still defeats the purpose, and the original review could only split.

* **`merge <id> <id> …`** folds clusters into the first one, with the merge recorded in
  `stats["step_4"]["interactive_merges"]` just like the splits. Merging a cluster that holds the
  link property is refused.
* **`candidates [n]`** ranks cluster pairs by the cosine similarity of their centroids, which
  surfaces the fragments without hunting for them. It reliably finds cases such as
  `paper:venue`/`publication_venue` (split by the technical prefix rather than by meaning) or
  `Dataset used`/`used datasets`.
* The review is a command loop (`largest`, `find`, `split`, `merge`, `candidates`, `done`)
  instead of a fixed question sequence, so clusters can be inspected in any order.

Splitting no longer loads the embedding model in the common case: the vectors from step 3 are
reused from cache, and only center terms that are not existing labels have to be encoded.

---

## Comparability with the paper

The resulting ontology is **not** the published v1.0.0. It is produced with a different chat
model and from a newer property snapshot, and it is versioned `1.1.0` to make that explicit.
Everything the paper reports refers to v1.0.0, which remains available unchanged under its own
version IRI.

The two models were compared before switching: on a balanced sample of 200 properties from the
original run (100 kept, 100 rejected by `glm-4.5-air`), the new model agreed with 82.5% of the
decisions. Inspecting the differences, most of them are cases where the new model follows the
written criteria of `prompts.yaml` more closely than the original run did — it rejects domain
variables such as `Si-O-Si bending` and `Sleep efficiency (%)` that the prompt explicitly lists
as invalid, and keeps relations such as `has ecosystem type` and metrics such as
`Best Mean Average Precision` that the prompt lists as valid.

That is an argument for the new model, but it is not a validation. The validation is the
evaluation protocol from the paper, which was re-run against v1.1.0.

## Evaluation of v1.1.0

Same gold standard, same embedding model, same threshold as in the paper.

| metric | v1.0.0 | v1.1.0 |
| --- | --- | --- |
| avg. search candidates, original ORKG | 47.44 | 45.53 |
| avg. search candidates, consolidated | 7.98 | 10.16 |
| ambiguity reduction | 83.2% | 77.7% |
| Wilcoxon signed-rank, *p* | 9.58 × 10⁻⁴¹ | 4.12 × 10⁻⁴¹ |
| provenance path coverage | 253/253 = 100% | 253/253 = 100% |

Both metrics hold: the reduction in semantic ambiguity remains highly significant, and every
gold-standard property still resolves to a canonical property, so no historical data is lost.

The reduction is lower than in v1.0.0 for two measured reasons, neither of which is a regression in
consolidation quality. v1.1.0 has 10.8% more canonical properties and clusters more conservatively,
deliberately keeping near-synonyms apart (§12). And v1.0.0's figure was flattered by the
normalisation defect described in §11: its 163 canonical labels containing underscores sat outside
natural-language space and were effectively invisible to the search metric.

One caveat for anyone re-running the evaluation: it counts embedding similarities against a hard
threshold and is therefore sensitive to the numerical precision of the embedding model. Loading
`Qwen3-Embedding-8B` in bfloat16 instead of float32 moves the headline metric for the *identical*
v1.0.0 ontology from 7.98 to 12.79. `Gold_Standard_evaluation.py` pins float32 and records the
environment in the results file; `requirements.txt` is pinned for the same reason.
