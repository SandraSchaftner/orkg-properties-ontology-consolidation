# Handover: publishing OPO-Consolidated v1.1.0

Read this if you are picking up the work in the
[orkg-properties-ontology-consolidation](https://github.com/SandraSchaftner/orkg-properties-ontology-consolidation)
repository. It says what was produced, why it differs from the published v1.0.0, and what is left
to do before v1.1.0 can be released.

Everything described here was built in a different repository
([rkg-templates](https://gitlab.com/TIBHannover/orkg/rkg-templates)), where the consolidated
ontology is used to resolve semantically identical ORKG properties before counting how often they
are used per research field. Producing v1.1.0 was a means to that end; publishing it as an
ontology release is the step that has not happened yet.

---

## What exists

| item | where |
| --- | --- |
| ontology v1.1.0, ready to publish | `opo-consolidated-1.1.0.ttl` and `.jsonld` (with version IRI) |
| pipeline that produced it | `opo/opo_consolidation.py`, `opo/fetch_properties.py`, `opo/prompts.yaml` |
| input snapshot | `opo/input/orkg_properties_2026-08-14.json` (+ `.meta.json` with timestamps) |
| checkpoints of all five steps | `opo/runs/current/checkpoints/` |
| what changed against the original pipeline | `opo/CHANGES.md` |

The run: 14,193 properties in, 3,175 canonical properties and 11,018 mappings out — the two add up
exactly to the input. Every mapping chain resolves to a canonical property (at most two hops, no
cycles, no dead ends), and 0 of roughly 35,000 LLM calls failed. Steps 1–3 took 3.97 h unattended,
the manual expert review 96 minutes (18 splits, 20 merges), step 5 another 17 minutes.

## How v1.1.0 differs from the published v1.0.0

Three differences matter for the release text; `opo/CHANGES.md` has the full list with measurements.

1. **A different chat model.** v1.0.0 used `glm-4.5-air` locally, v1.1.0 uses
   `Qwen3.6-35B-A3B-MLX-8bit` on the VSR inference server. On a balanced sample of 200 properties
   from the v1.0.0 run the two models agreed on 82.5%. Inspecting the differences, the new model
   follows the written criteria of `prompts.yaml` more closely in most cases — it rejects domain
   variables such as `Si-O-Si bending` and `Sleep efficiency (%)` that the prompt lists as invalid,
   and keeps relations such as `has ecosystem type` that it lists as valid. It is also stricter
   overall: 49% of properties rejected versus 40%. The embedding model is unchanged.
2. **A newer input snapshot**: 2026-08-14 instead of 2025-12-31, 14,193 instead of about 12,700
   properties. Notably `P4077` ("source code"), the single most used predicate in ORKG with 112k
   statements, was not covered by v1.0.0 at all and is now.
3. **The link property is kept out of the clustering.** `P41267` collects everything step 2
   rejects. Carrying a label like any other property, it took part in the clustering itself and in
   v1.0.0 ended up merged into "links" (`P41267 → P15325`). Resolving a mapping chain to its end
   therefore returned a legitimate property for every rejected one. In v1.1.0 it is set aside
   before clustering, excluded from normalisation, and takes no part in label collisions. This is
   the only change that alters the *result* rather than the execution.

## What is left to do

### 1. Re-run the evaluation

`Gold_Standard_evaluation.py` in the ontology repo implements the protocol from the paper
(ambiguity analysis and provenance path validation). It has not been run against v1.1.0. Points to
watch:

- `CONSOLIDATED_FILE` must point at the new JSON-LD.
- `ORIGINAL_PROPERTIES_FILE` should be the snapshot v1.1.0 was actually built from —
  `orkg_properties_2026-08-14.json` — not the 2025-12-31 export. Comparing the new ontology against
  the old original would mix two states of ORKG.
- The gold standard (`orkg_properties_llm_dimensions_dataset(1).csv`) and the embedding model stay
  as they are, otherwise the numbers are not comparable with the paper.
- Expect the ambiguity numbers to differ from the paper: v1.1.0 has 3,175 canonical properties
  against 2,865, from a larger input.

### 2. Version IRI — already done

The artifacts to publish carry both triples:

```turtle
owl:versionIRI <https://w3id.org/orkg-properties-ontology-consolidated/1.1.0> ;
owl:priorVersion <https://w3id.org/orkg-properties-ontology-consolidated/1.0.0> ;
```

so the two releases are distinguishable as RDF, not only by file name, and the new one points back
at what it supersedes. They were generated on 2026-08-17 from the unchanged step 4 checkpoint —
same 3,175 canonical properties and 11,018 mappings as before, only the header differs.

### 3. Point the w3id URI at v1.1.0 — after the evaluation

`https://w3id.org/orkg-properties-ontology-consolidated` currently resolves to the v1.0.0 Turtle
file. The paper is published and presented, so the redirect should move to **v1.1.0** once the
evaluation confirms the new release. Add permanent version-specific redirects alongside it
(`/1.0.0`, `/1.1.0`) — the version IRIs above already assume that shape — and state in the README
which version the paper describes, so a reader of the paper can still reach the artifact it
evaluated. Do the switch only after the evaluation: the URI should never point at an unevaluated
artifact.

### 4. Adapt the documentation

- `opo/CHANGES.md` is written from the perspective of the analysis project ("preparatory work for
  the rkg-templates analysis"). For the ontology repository it should be reframed as release notes
  for v1.1.0 — the content stays, the framing changes.
- The repository README needs a section saying that v1.1.0 exists, how it differs, which version
  the paper describes, and where the code that produced it lives.
- `opo/README.md` describes how to run the pipeline and can move over unchanged.

## How to work in this repository

Work directly on `main`, in small commits. A feature branch would only add a merge step: there is
one author, no review and no CI. If a branch is used anyway, it has to be merged when the release
is finished — a release sitting on a side branch is invisible, the README would reference files
that `main` does not have, and a w3id redirect pointing at it would resolve to nothing.

**Do not move, rename or delete `ontology/opo-consolidated-1.0.0.ttl`.** The w3id URI resolves to
that path and the published paper cites it. Everything new goes next to it, so the old URI keeps
working throughout.

When the evaluation is done and the documentation is updated, mark the state with a tag rather
than a branch:

```bash
git tag -a v1.1.0 -m "OPO-Consolidated 1.1.0"
git push origin v1.1.0
```

That makes the exact state permanently referenceable and can be turned into a GitHub release with
a description. If a DOI for the version is wanted, that release is also what Zenodo hooks into.

## Things that will trip you up

- **`opo/runs/current/cache/` is about 49 MB** of LLM decision caches. It is deliberately not
  versioned. If the folder is copied with a file manager rather than with git, the caches come
  along — delete them or copy only what git tracks.
- **The API key** for the inference server lives in `.env` in the analysis repository, which is
  gitignored. It is a temporary VSR test key; if a run fails with 401, that is why.
- **Do not re-run steps 1–4** to reproduce the artifacts. The manual review is 96 minutes of expert
  judgement that is stored in `opo/runs/current/checkpoints/step_4_final/`; re-running the earlier
  steps would invalidate it. `--start-step 5` is enough for any change to the export.
- **The evaluation needs `torch` and `transformers`** for the embedding model, about 3 GB, plus the
  14 GB model itself in the Hugging Face cache.
