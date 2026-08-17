# Step 4 review guide: your v1.0.0 splitting decisions in the new clustering

For each cluster you split by hand in the v1.0.0 run, this lists where its labels ended up
in the new clustering (`opo/runs/current`, 2026-08-15) and whether the split is still needed.

Labels in **bold** are the ones from your old cluster; the others joined the new cluster since.
Labels missing entirely were rejected in step 2 of the new run (mapped to `P41267`).

Generated from `checkpoints/step_4_final/stats.json` of the ontology repo.

## Overview

- **14** of 51 old decisions: cluster is still together, split still needed
- **35**: the new clustering already separated the labels — check only
- **2**: all labels were rejected in step 2, nothing left to do

| old cluster | labels then | still present | status | new cluster(s) |
| --- | --- | --- | --- | --- |
| 0 | 43 | 33 | ALREADY SEPARATED | `0`, `1`, `2`, `3`, `4`, `5` … |
| 1 | 40 | 38 | ALREADY SEPARATED | `24`, `25`, `26`, `27`, `28`, `29` … |
| 3 | 25 | 24 | ALREADY SEPARATED | `99`, `100`, `101`, `102`, `103`, `104` … |
| 8 | 20 | 17 | ALREADY SEPARATED | `46`, `47`, `48`, `50`, `52`, `55` … |
| 9 | 20 | 18 | ALREADY SEPARATED | `122`, `123`, `124`, `125`, `126`, `127` … |
| 12 | 17 | 17 | ALREADY SEPARATED | `91`, `93`, `137`, `139`, `141`, `706` … |
| 16 | 16 | 16 | ALREADY SEPARATED | `111`, `401` |
| 20 | 15 | 12 | ALREADY SEPARATED | `267`, `268`, `269`, `270`, `271`, `1748` |
| 17 | 15 | 10 | SPLIT STILL NEEDED | `210` |
| 23 | 14 | 12 | ALREADY SEPARATED | `249`, `250`, `251`, `252`, `253`, `254` … |
| 35 | 13 | 12 | ALREADY SEPARATED | `163`, `164`, `165`, `166`, `167` |
| 46 | 12 | 4 | ALREADY SEPARATED | `967`, `968`, `969` |
| 47 | 12 | 11 | ALREADY SEPARATED | `232`, `2135` |
| 40 | 12 | 11 | SPLIT STILL NEEDED | `135` |
| 61 | 11 | 7 | ALREADY SEPARATED | `615`, `617`, `618`, `619`, `1149`, `1997` |
| 54 | 11 | 10 | SPLIT STILL NEEDED | `201` |
| 59 | 11 | 10 | ALREADY SEPARATED | `305`, `306`, `307`, `308`, `309`, `359` |
| 56 | 11 | 9 | ALREADY SEPARATED | `515`, `516`, `1512` |
| 50 | 11 | 11 | ALREADY SEPARATED | `183`, `530`, `873`, `1881` |
| 52 | 11 | 11 | ALREADY SEPARATED | `202`, `203`, `204`, `205`, `206`, `207` … |
| 60 | 11 | 11 | ALREADY SEPARATED | `157`, `215`, `791`, `792` |
| 68 | 10 | 9 | ALREADY SEPARATED | `258`, `391`, `392`, `393`, `394`, `578` |
| 75 | 10 | 9 | ALREADY SEPARATED | `293`, `294`, `295`, `296`, `297`, `298` … |
| 103 | 9 | 8 | ALREADY SEPARATED | `541`, `1369` |
| 102 | 9 | 9 | ALREADY SEPARATED | `217`, `218`, `219`, `220`, `221` |
| 89 | 9 | 9 | ALREADY SEPARATED | `496`, `497`, `499`, `511`, `512`, `513` |
| 114 | 8 | 8 | ALREADY SEPARATED | `330`, `331`, `332`, `333`, `334` |
| 141 | 8 | 5 | SPLIT STILL NEEDED | `655` |
| 122 | 8 | 6 | ALREADY SEPARATED | `659`, `661`, `662`, `663`, `1558` |
| 113 | 8 | 8 | SPLIT STILL NEEDED | `449` |
| 125 | 8 | 8 | SPLIT STILL NEEDED | `338` |
| 133 | 8 | 8 | ALREADY SEPARATED | `258`, `452` |
| 128 | 8 | 8 | ALREADY SEPARATED | `291`, `1353` |
| 109 | 8 | 7 | ALREADY SEPARATED | `90`, `92`, `94`, `97`, `98` |
| 119 | 8 | 8 | ALREADY SEPARATED | `347`, `348`, `349`, `350`, `351`, `352` |
| 171 | 7 | 6 | ALREADY SEPARATED | `234`, `235`, `237`, `238`, `241` |
| 191 | 7 | 7 | SPLIT STILL NEEDED | `259` |
| 157 | 7 | 7 | SPLIT STILL NEEDED | `468` |
| 274 | 5 | 0 | GONE | — |
| 264 | 5 | 4 | SPLIT STILL NEEDED | `824` |
| 339 | 5 | 2 | ALREADY SEPARATED | `1384`, `1595` |
| 318 | 5 | 3 | ALREADY SEPARATED | `1233`, `1234` |
| 301 | 5 | 4 | ALREADY SEPARATED | `753`, `755`, `756` |
| 452 | 4 | 2 | SPLIT STILL NEEDED | `443` |
| 473 | 4 | 1 | SPLIT STILL NEEDED | `3075` |
| 387 | 4 | 3 | ALREADY SEPARATED | `1060`, `1061` |
| 541 | 4 | 0 | GONE | — |
| 720 | 3 | 3 | SPLIT STILL NEEDED | `1161` |
| 670 | 3 | 3 | ALREADY SEPARATED | `1086`, `1087` |
| 810 | 3 | 2 | SPLIT STILL NEEDED | `1033` |
| 1601 | 2 | 2 | SPLIT STILL NEEDED | `2041` |

---

---

## old cluster 0 — ALREADY SEPARATED

*Your centers:* `uses ontology, ontology name, ontology iri`

43 labels then, 33 of them still present (10 rejected in step 2), now in 17 cluster(s).

The new clustering already pulled these apart:

- **cluster `0`** (2 labels) contains 2 of yours
  > **ontology design pattern**, **Schema ontology**
- **cluster `1`** (7 labels) contains 7 of yours
  > **Ontologies used**, **Ontology used**, **Used Ontologies**, **Used Upper Ontologies**, **Uses ontology**, **uses_ontolology**, **using an ontology**
- **cluster `2`** (2 labels) contains 2 of yours
  > **Ontology IRI**, **Ontology URL**
- **cluster `3`** (2 labels) contains 2 of yours
  > **Ontology name**, **Ontology type**
- **cluster `4`** (3 labels) contains 3 of yours
  > **Ontologies**, **Ontology**, **Ontology domains**
- **cluster `5`** (3 labels) contains 1 of yours
  > **Ontology construction**, Ontology Construction Method, Ontology Development Methodology
- **cluster `7`** (1 labels) contains 1 of yours
  > **ontology editor**
- **cluster `10`** (3 labels) contains 3 of yours
  > **om ontology**, **prov ontology**, **qu ontology**
- **cluster `11`** (3 labels) contains 3 of yours
  > **ontology based**, **ontology component**, **ontology module**
- **cluster `12`** (1 labels) contains 1 of yours
  > **Linked Ontology**
- **cluster `16`** (1 labels) contains 1 of yours
  > **Has ontology**
- **cluster `17`** (1 labels) contains 1 of yours
  > **Ontology Language**
- **cluster `18`** (1 labels) contains 1 of yours
  > **Ontology ID**
- **cluster `20`** (1 labels) contains 1 of yours
  > **Ontology Orientation**
- **cluster `21`** (1 labels) contains 1 of yours
  > **Ontology availability**
- **cluster `23`** (2 labels) contains 2 of yours
  > **reused ontologies**, **Reused ontology**
- **cluster `3084`** (1 labels) contains 1 of yours
  > **Number of ontologies referenced**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Available Ontology Languages, Described Ontology, Existing ontologies, Ontology file availability, OntologyName, Reused Ontologies Ratio, Reused Ontologies Ratio (%), Upper-level ontology, ontology retrieval, referenced ontology

---

## old cluster 1 — ALREADY SEPARATED

*Your centers:* `evaluation dataset, evaluation metric, evaluation method, evaluation results, evaluation criteria, evaluation benchmark`

40 labels then, 38 of them still present (2 rejected in step 2), now in 19 cluster(s).

The new clustering already pulled these apart:

- **cluster `24`** (4 labels) contains 4 of yours
  > **evaluation**, **evaluation aspect**, **evaluation item**, **Evaluator**
- **cluster `25`** (4 labels) contains 4 of yours
  > **evaluation measures**, **evaluation method**, **Evaluation methods**, **Means of evaluation**
- **cluster `26`** (6 labels) contains 6 of yours
  > **Evaluation metics**, **Evaluation metric**, **Evaluation metrics**, **evaluation_metric**, **evaluation_metrics**, **Has evaluation metrics**
- **cluster `27`** (2 labels) contains 1 of yours
  > evaluated against, **evaluates**
- **cluster `28`** (3 labels) contains 3 of yours
  > **evaluate on**, **evaluate-on**, **Evaluated on**
- **cluster `29`** (1 labels) contains 1 of yours
  > **evaluation_reporting**
- **cluster `30`** (2 labels) contains 2 of yours
  > **Eval. of results**, **Evaluation results**
- **cluster `33`** (2 labels) contains 1 of yours
  > **evaluation benchmark**, Evaluation Benchmarks
- **cluster `34`** (2 labels) contains 2 of yours
  > **evaluation_framework**, **evaluation_frameworks**
- **cluster `35`** (1 labels) contains 1 of yours
  > **evaluation_phases**
- **cluster `36`** (1 labels) contains 1 of yours
  > **analytical_evaluation**
- **cluster `37`** (1 labels) contains 1 of yours
  > **in-situ_evaluation**
- **cluster `39`** (1 labels) contains 1 of yours
  > **empirical_evaluation**
- **cluster `40`** (1 labels) contains 1 of yours
  > **computational evaluation**
- **cluster `42`** (1 labels) contains 1 of yours
  > **evaluation setting**
- **cluster `44`** (2 labels) contains 2 of yours
  > **evaluation criteria**, **evaluation_criteria_selection**
- **cluster `45`** (1 labels) contains 1 of yours
  > **evaluation_resources**
- **cluster `358`** (9 labels) contains 4 of yours
  > benchmark dataset, benchmark_dataset, benchmarked datasets, **evaluation data**, **evaluation data split**, **evaluation dataset**, **Evaluation Datasets**, evalutation dataset, On evaluation dataset
- **cluster `1920`** (2 labels) contains 1 of yours
  > **evaluation_dimensions**, scoring dimensions

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* evaluate, evaluation_execution

---

## old cluster 3 — ALREADY SEPARATED

*Your centers:* `dataset, dataset url`

25 labels then, 24 of them still present (1 rejected in step 2), now in 13 cluster(s).

The new clustering already pulled these apart:

- **cluster `99`** (3 labels) contains 3 of yours
  > **existing datasets utilization**, **used datasets**, **utilized datasets**
- **cluster `100`** (3 labels) contains 2 of yours
  > Data repositories, **data_repository**, **dataset_repository**
- **cluster `101`** (3 labels) contains 2 of yours
  > **Datasets**, **Has Datasets**, No. of datasets
- **cluster `102`** (2 labels) contains 2 of yours
  > **Dataset name**, **Dataset_name/ source**
- **cluster `103`** (2 labels) contains 2 of yours
  > **Dataset used**, **Datasets used**
- **cluster `104`** (1 labels) contains 1 of yours
  > **Dataset location**
- **cluster `105`** (1 labels) contains 1 of yours
  > **Dataset statistics**
- **cluster `106`** (2 labels) contains 2 of yours
  > **dataset characteristics**, **dataset features**
- **cluster `107`** (2 labels) contains 1 of yours
  > data, **dataset**
- **cluster `108`** (1 labels) contains 1 of yours
  > **dataset description**
- **cluster `109`** (1 labels) contains 1 of yours
  > **Other datasets**
- **cluster `597`** (6 labels) contains 5 of yours
  > contribution_experimental_dataset, **dataset contruction**, **Experimental Dataset**, **Has experimental datasets**, **Proposed Dataset**, **proposed_dataset**
- **cluster `1072`** (3 labels) contains 1 of yours
  > data_repo_url, **Dataset download url**, Dataset URL

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* data_set

---

## old cluster 8 — ALREADY SEPARATED

*Your centers:* `experiment, experiment condition, place of experiment`

20 labels then, 17 of them still present (3 rejected in step 2), now in 11 cluster(s).

The new clustering already pulled these apart:

- **cluster `46`** (2 labels) contains 1 of yours
  > experiment id, **Experiment name**
- **cluster `47`** (5 labels) contains 3 of yours
  > **experiment**, **experiment on**, experiment type, experiment with, **type of experiment**
- **cluster `48`** (3 labels) contains 1 of yours
  > experiment description, experiment details, **Experimental details**
- **cluster `50`** (2 labels) contains 1 of yours
  > experiment location, **Place of experiment**
- **cluster `52`** (2 labels) contains 2 of yours
  > **experiment condition**, **Experimental conditions**
- **cluster `55`** (3 labels) contains 3 of yours
  > **Experiment environment setting**, **Experiment setting**, **Experimental Setup**
- **cluster `56`** (1 labels) contains 1 of yours
  > **Experiment approach**
- **cluster `57`** (1 labels) contains 1 of yours
  > **Experiment phases**
- **cluster `58`** (1 labels) contains 1 of yours
  > **experiment series**
- **cluster `63`** (1 labels) contains 1 of yours
  > **Experiment queries**
- **cluster `2154`** (2 labels) contains 2 of yours
  > **Experimental Design**, **has experimental design**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Experiment 1, Experiment 2, Experimental Settings

---

## old cluster 9 — ALREADY SEPARATED

*Your centers:* `observation data, observation technique`

20 labels then, 18 of them still present (2 rejected in step 2), now in 12 cluster(s).

The new clustering already pulled these apart:

- **cluster `122`** (5 labels) contains 5 of yours
  > **observation data**, **observation number**, **observation type**, **observation type description**, **observation_data**
- **cluster `123`** (2 labels) contains 2 of yours
  > **is direct observation**, **Observation**
- **cluster `124`** (2 labels) contains 2 of yours
  > **observation_technique**, **observation_techniques**
- **cluster `125`** (1 labels) contains 1 of yours
  > **Observed value**
- **cluster `126`** (1 labels) contains 1 of yours
  > **observes**
- **cluster `127`** (1 labels) contains 1 of yours
  > **observed feature**
- **cluster `128`** (1 labels) contains 1 of yours
  > **Observer**
- **cluster `129`** (1 labels) contains 1 of yours
  > **Observed in**
- **cluster `130`** (1 labels) contains 1 of yours
  > **manual_observation**
- **cluster `131`** (1 labels) contains 1 of yours
  > **observation unit**
- **cluster `132`** (1 labels) contains 1 of yours
  > **has observations**
- **cluster `1183`** (3 labels) contains 1 of yours
  > **Has observation period**, observation_duration, observation_period

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* observe, observer_effect

---

## old cluster 12 — ALREADY SEPARATED

*Your centers:* `publication venue, publication date`

17 labels then, 17 of them still present (0 rejected in step 2), now in 7 cluster(s).

The new clustering already pulled these apart:

- **cluster `91`** (6 labels) contains 5 of yours
  > **paper: publication_year**, **paper:publication_year**, **paper:publicationDate**, **paper:puplication_year**, **paper:year**, paper_year
- **cluster `93`** (5 labels) contains 4 of yours
  > **paper:journal**, **paper:publicationTypes**, paper:publised_in, **paper:published_in**, **paper:venue**
- **cluster `137`** (3 labels) contains 2 of yours
  > **Publication type**, Publication type Period, **PublicationTypes**
- **cluster `139`** (1 labels) contains 1 of yours
  > **publication_venue**
- **cluster `141`** (1 labels) contains 1 of yours
  > **Publication location**
- **cluster `706`** (3 labels) contains 3 of yours
  > **journal metadata**, **publication-info metadata**, **publisher metadata**
- **cluster `2053`** (2 labels) contains 1 of yours
  > Bibliographic data source, **BibliographyType**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 16 — ALREADY SEPARATED

*Your centers:* `application, application field`

16 labels then, 16 of them still present (0 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `111`** (20 labels) contains 15 of yours
  > **App. Type**, **application**, **Application area**, **application category**, **Application Domain**, **application element**, **Application field**, **Application field(s)**, **application sub category**, **Application type**, **application used**, **application_element**, **application_type**, applicationCategory, Applications, applicationSuite, Domain / Application Area, **domain of application**, **Field of application**, has application domain
- **cluster `401`** (8 labels) contains 1 of yours
  > **Application area studied**, Area of study, has research domain, Research Area, research field, research_area, research_field, research_field_investigated

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 20 — ALREADY SEPARATED

*Your centers:* `pretraining model, pretraining task`

15 labels then, 12 of them still present (3 rejected in step 2), now in 6 cluster(s).

The new clustering already pulled these apart:

- **cluster `267`** (3 labels) contains 3 of yours
  > **trained for**, **trained-on**, **transformer trained on**
- **cluster `268`** (4 labels) contains 4 of yours
  > **pre-trained for**, **pre-trained model used**, **pre-trained on**, **Pretrained model**
- **cluster `269`** (1 labels) contains 1 of yours
  > **trained parameter**
- **cluster `270`** (2 labels) contains 2 of yours
  > **Pre-training task(s)**, **pretraining task**
- **cluster `271`** (1 labels) contains 1 of yours
  > **pretraining architecture**
- **cluster `1748`** (2 labels) contains 1 of yours
  > **pretraining corpus**, size of training corpus (in tokens in billions)

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Pre-train then Fine-tune, pre-trained, pre-training language model

---

## old cluster 17 — SPLIT STILL NEEDED

*Your centers:* `baseline result, baseline`

15 labels then, 10 of them still present (5 rejected in step 2), now in 1 cluster(s).

**→ new cluster `210`** (12 labels)

> **Baseline**, **Baseline comparison**, **Baseline comparison detail**, **Baseline comparison type**, Baseline Comparisons, **baseline result**, **Baseline score**, **baseline_measurement**, Comparison Baselines, **has base line**, **has baseline**, **has human baseline**

Command in step 4: inspect `210`, split, and paste:

```
baseline result, baseline
```

*Rejected in step 2:* Baseline results, Has-baseline, baseline evaluation, baseline_SER, has baselines

---

## old cluster 23 — ALREADY SEPARATED

*Your centers:* `benchmark corpus, corpus, corpus source`

14 labels then, 12 of them still present (2 rejected in step 2), now in 10 cluster(s).

The new clustering already pulled these apart:

- **cluster `249`** (3 labels) contains 3 of yours
  > **corpus origin**, **corpus source**, **Corpus sources**
- **cluster `250`** (1 labels) contains 1 of yours
  > **Testing corpus**
- **cluster `251`** (1 labels) contains 1 of yours
  > **corpus volume**
- **cluster `252`** (1 labels) contains 1 of yours
  > **Finetuning corpus**
- **cluster `253`** (1 labels) contains 1 of yours
  > **Corpus statistics**
- **cluster `254`** (1 labels) contains 1 of yours
  > **Corpus Cardinality**
- **cluster `255`** (1 labels) contains 1 of yours
  > **Corpus name**
- **cluster `256`** (1 labels) contains 1 of yours
  > **Corpora**
- **cluster `257`** (1 labels) contains 1 of yours
  > **Corpus tailored**
- **cluster `2019`** (1 labels) contains 1 of yours
  > **gold-standard corpora**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Benchmark Corpus, corpus tiers

---

## old cluster 35 — ALREADY SEPARATED

*Your centers:* `ACM Digital Library author ID, Arnet Miner author ID, Author(s) ID, DBLP author ID, Dimensions author ID, Google Scholar ID, ORCID, OpenAlex ID, Publons ID, ResearchGate ID, ResearcherID, Scopus author ID, Semantic Scholar author ID`

13 labels then, 12 of them still present (1 rejected in step 2), now in 5 cluster(s).

The new clustering already pulled these apart:

- **cluster `163`** (14 labels) contains 2 of yours
  > author, **Author(s) ID**, AuthorCount, authors, Authors: Country, authorUnordered, **Dimensions author ID**, First author, first_author, FirstAuthor, has author, Number of authors, paper: authors, paper:author
- **cluster `164`** (6 labels) contains 6 of yours
  > **ACM Digital Library author ID**, **DBLP author ID**, **Google Scholar ID**, **OpenAlex ID**, **Scopus author ID**, **Semantic Scholar author ID**
- **cluster `165`** (5 labels) contains 1 of yours
  > Has ORCID, has ORCID ID, hasORCID, **ORCID**, orcidId
- **cluster `166`** (2 labels) contains 2 of yours
  > **ResearcherID**, **ResearchGate ID**
- **cluster `167`** (1 labels) contains 1 of yours
  > **Publons ID**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Arnet Miner author ID

---

## old cluster 46 — ALREADY SEPARATED

*Your centers:* `ethical approval, ethical documentation`

12 labels then, 4 of them still present (8 rejected in step 2), now in 3 cluster(s).

The new clustering already pulled these apart:

- **cluster `967`** (2 labels) contains 2 of yours
  > **ethical_approval**, **ethics_committee_approval**
- **cluster `968`** (1 labels) contains 1 of yours
  > **ethical_guideline**
- **cluster `969`** (1 labels) contains 1 of yours
  > **ethical implication**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* community_ethical_feedback, ethical_consideration, ethical_dimension, ethical_documentation, ethical_implication, ethical_reflection, ethics considerations, evaluation_ethics

---

## old cluster 47 — ALREADY SEPARATED

*Your centers:* `GitHub, code repository`

12 labels then, 11 of them still present (1 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `232`** (11 labels) contains 9 of yours
  > **code**, code available, **Code repositories**, **code repository**, **code repository (compiled)**, code_availability, **code_repo_url**, **code_repository**, **source code**, **source code repository URL**, **Sourcecode**
- **cluster `2135`** (2 labels) contains 2 of yours
  > **Github**, **Github link**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Code respositories

---

## old cluster 40 — SPLIT STILL NEEDED

*Your centers:* `threats to validity, internal validity, external validity, construct validity, conclusion validity, validity`

12 labels then, 11 of them still present (1 rejected in step 2), now in 1 cluster(s).

**→ new cluster `135`** (17 labels)

> **conclusion validity**, confirmability validity, **construct validity**, construct_validity, **content validity**, **descriptive validity**, ecological_validity_measure, **external validity**, **internal validity**, **internal_validity**, **research_validity**, **theoretical validity**, **threat to validity**, **Threats To Validity**, validity_dimension, validity_measure, validity_measures

Command in step 4: inspect `135`, split, and paste:

```
threats to validity, internal validity, external validity, construct validity, conclusion validity, validity
```

*Rejected in step 2:* threat to construct validity

---

## old cluster 61 — ALREADY SEPARATED

*Your centers:* `learning model, learning method`

11 labels then, 7 of them still present (4 rejected in step 2), now in 6 cluster(s).

The new clustering already pulled these apart:

- **cluster `615`** (2 labels) contains 2 of yours
  > **Learning approach**, **Learning method**
- **cluster `617`** (1 labels) contains 1 of yours
  > **Learning purpose**
- **cluster `618`** (1 labels) contains 1 of yours
  > **learning model**
- **cluster `619`** (1 labels) contains 1 of yours
  > **learning technique**
- **cluster `1149`** (2 labels) contains 1 of yours
  > **learning paradigm**, Training Paradigm
- **cluster `1997`** (1 labels) contains 1 of yours
  > **Class learning**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Being taught, Learning, learn, user_learning_process

---

## old cluster 54 — SPLIT STILL NEEDED

*Your centers:* `benchmark, benchmark result, benchmark model`

11 labels then, 10 of them still present (1 rejected in step 2), now in 1 cluster(s).

**→ new cluster `201`** (12 labels)

> **Benchmark**, **Benchmark Description**, **benchmark name**, **Benchmark Performance**, benchmark results, **Benchmark Time**, **benchmark_model**, benchmark_result, **benchmarking**, **has benchmark**, **Total benchmarks**, **uses benchmark**

Command in step 4: inspect `201`, split, and paste:

```
benchmark, benchmark result, benchmark model
```

*Rejected in step 2:* contains benchmark

---

## old cluster 59 — ALREADY SEPARATED

*Your centers:* `performance, performance metric, performance result`

11 labels then, 10 of them still present (1 rejected in step 2), now in 6 cluster(s).

The new clustering already pulled these apart:

- **cluster `305`** (4 labels) contains 3 of yours
  > Key Performance Metrics, **Performance measures**, **Performance metric**, **Performance metrics**
- **cluster `306`** (2 labels) contains 1 of yours
  > **Performance criteria**, performance evaluation criteria
- **cluster `307`** (2 labels) contains 1 of yours
  > Key Performance Results, **Performance Results**
- **cluster `308`** (1 labels) contains 1 of yours
  > **Number of performance measures**
- **cluster `309`** (1 labels) contains 1 of yours
  > **functional performance measure**
- **cluster `359`** (9 labels) contains 3 of yours
  > Has (Best )performance, has performance, **performance**, **Performance measurment**, performance validation, Performance:Precision, Performance:Specificity, **performance_type**, Reported performance

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* performance_and_usage_metrics

---

## old cluster 56 — ALREADY SEPARATED

*Your centers:* `evaluation task, evaluation tool, evaluation result, recall, specificity`

11 labels then, 9 of them still present (2 rejected in step 2), now in 3 cluster(s).

The new clustering already pulled these apart:

- **cluster `515`** (2 labels) contains 2 of yours
  > **evaluated task**, **Has evaluation task**
- **cluster `516`** (5 labels) contains 5 of yours
  > **Has evaluation**, **has evaluation result**, **has evaluation score**, **has evaluation tool**, **hasEvaluation**
- **cluster `1512`** (2 labels) contains 2 of yours
  > **has evaluation result-recall**, **has result-recall**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* has evaluation result-specificity, has evaluation results-recall

---

## old cluster 50 — ALREADY SEPARATED

*Your centers:* `related to, is about`

11 labels then, 11 of them still present (0 rejected in step 2), now in 4 cluster(s).

The new clustering already pulled these apart:

- **cluster `183`** (14 labels) contains 7 of yours
  > **associated with**, Has Relations, **related to**, **related_to**, **relatedTo**, **relates from**, **relates to**, **relation**, Relation representation, relation type, Relation1, Relation2, relation_type, sub-relations
- **cluster `530`** (1 labels) contains 1 of yours
  > **related domain**
- **cluster `873`** (2 labels) contains 2 of yours
  > **aligned to**, **aligned with**
- **cluster `1881`** (2 labels) contains 1 of yours
  > deals with, **is about**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 52 — ALREADY SEPARATED

*Your centers:* `deployment, deployment location, deployment log`

11 labels then, 11 of them still present (0 rejected in step 2), now in 7 cluster(s).

The new clustering already pulled these apart:

- **cluster `202`** (2 labels) contains 2 of yours
  > **deployment_period**, **deployment_phase**
- **cluster `203`** (2 labels) contains 2 of yours
  > **Deployment location**, **deployment_area**
- **cluster `204`** (2 labels) contains 2 of yours
  > **deployed in**, **deployed_at_location**
- **cluster `205`** (1 labels) contains 1 of yours
  > **deployment_log**
- **cluster `206`** (1 labels) contains 1 of yours
  > **deployment_status**
- **cluster `207`** (1 labels) contains 1 of yours
  > **deployment_context**
- **cluster `208`** (2 labels) contains 2 of yours
  > **Deploymen**, **Deployment**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 60 — ALREADY SEPARATED

*Your centers:* `citation, cites`

11 labels then, 11 of them still present (0 rejected in step 2), now in 4 cluster(s).

The new clustering already pulled these apart:

- **cluster `157`** (16 labels) contains 1 of yours
  > ** Cites / Source**, # Sources, Claim Sources, has relevant sources, has source, has sources, hasSource, Information sources, Other sources/ (papers), source, source authority, Source domain, source material, Source name, source of material, Sources
- **cluster `215`** (12 labels) contains 7 of yours
  > # of References, **citation**, **Citations**, **cited**, **Cited by**, **cites work**, **Has Citations**, **isCitedBy**, number of references, reference, reference publication, References
- **cluster `791`** (2 labels) contains 2 of yours
  > **CitationURL**, **CitesURL**
- **cluster `792`** (1 labels) contains 1 of yours
  > **reference URL**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 68 — ALREADY SEPARATED

*Your centers:* `tool, tool URL`

10 labels then, 9 of them still present (1 rejected in step 2), now in 6 cluster(s).

The new clustering already pulled these apart:

- **cluster `258`** (11 labels) contains 1 of yours
  > Approach name, Framework / System Name, Name of system, Prototype System Name, System / Approach Name, System / Tool Name, system name, system_name, system_name_reference, Tool / System Name, **Tool name**
- **cluster `391`** (3 labels) contains 2 of yours
  > **tool  description**, **tool URL**, tool useses AI
- **cluster `392`** (2 labels) contains 2 of yours
  > **Available tools**, **Availible tools**
- **cluster `393`** (2 labels) contains 2 of yours
  > **Tool**, **Tools**
- **cluster `394`** (1 labels) contains 1 of yours
  > **tool availability**
- **cluster `578`** (6 labels) contains 1 of yours
  > hasEmployedTool, supported by tool, **tool type**, Tool-Support, uses tool, uses_tool

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Tool Findings

---

## old cluster 75 — ALREADY SEPARATED

*Your centers:* `project, project URL, project type`

10 labels then, 9 of them still present (1 rejected in step 2), now in 7 cluster(s).

The new clustering already pulled these apart:

- **cluster `293`** (3 labels) contains 3 of yours
  > **project**, **project end**, **project_name**
- **cluster `294`** (1 labels) contains 1 of yours
  > **Project type**
- **cluster `295`** (1 labels) contains 1 of yours
  > **project URL**
- **cluster `296`** (1 labels) contains 1 of yours
  > **project start**
- **cluster `297`** (1 labels) contains 1 of yours
  > **project state**
- **cluster `298`** (1 labels) contains 1 of yours
  > **Project title**
- **cluster `299`** (1 labels) contains 1 of yours
  > **Project progress**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* project_manager

---

## old cluster 103 — ALREADY SEPARATED

*Your centers:* `machine learning algorithm, machine learning model, machine learning framework`

9 labels then, 8 of them still present (1 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `541`** (7 labels) contains 6 of yours
  > **machine learning algorithm**, **Machine learning algorithms**, **machine learning algorithms/methods**, **machine learning models**, **Machine/Deep Learning Algorithms**, **ML algorithm**, Number of machine learning algorithms
- **cluster `1369`** (3 labels) contains 2 of yours
  > **Machine learning framework**, **machine learning framework used**, Machine Learning Paradigm

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Machine learning feature

---

## old cluster 102 — ALREADY SEPARATED

*Your centers:* `field of knowledge, knowledge source`

9 labels then, 9 of them still present (0 rejected in step 2), now in 5 cluster(s).

The new clustering already pulled these apart:

- **cluster `217`** (3 labels) contains 3 of yours
  > **domain knowledge**, **knowledge type**, **knowledge used**
- **cluster `218`** (4 labels) contains 2 of yours
  > **Knowledge source**, knowledge_source, knowledge_source_reference, **Type of knowledge source**
- **cluster `219`** (1 labels) contains 1 of yours
  > **Knowledge Base**
- **cluster `220`** (2 labels) contains 2 of yours
  > **Field of knowledge**, **Fielf of knowledge**
- **cluster `221`** (1 labels) contains 1 of yours
  > **Knowledge derived from**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 89 — ALREADY SEPARATED

*Your centers:* `NLP dataset, NLP task`

9 labels then, 9 of them still present (0 rejected in step 2), now in 6 cluster(s).

The new clustering already pulled these apart:

- **cluster `496`** (3 labels) contains 3 of yours
  > **NLP task**, **NLP task type**, **NLP tasks**
- **cluster `497`** (1 labels) contains 1 of yours
  > **NLP task input**
- **cluster `499`** (1 labels) contains 1 of yours
  > **Downstream NLP Task **
- **cluster `511`** (2 labels) contains 2 of yours
  > **NLP data format**, **NLP dataset**
- **cluster `512`** (2 labels) contains 1 of yours
  > **NLP data source**, NLP data source domain
- **cluster `513`** (2 labels) contains 1 of yours
  > NLP data source type, **NLP data type**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 114 — ALREADY SEPARATED

*Your centers:* `classification method, classification result, classification scheme`

8 labels then, 8 of them still present (0 rejected in step 2), now in 5 cluster(s).

The new clustering already pulled these apart:

- **cluster `330`** (2 labels) contains 2 of yours
  > **Classification scheme**, **classification system**
- **cluster `331`** (2 labels) contains 2 of yours
  > **Classification**, **Classification category**
- **cluster `332`** (3 labels) contains 2 of yours
  > **classification type**, **classifier**, classifier type
- **cluster `333`** (1 labels) contains 1 of yours
  > **Classification Result**
- **cluster `334`** (1 labels) contains 1 of yours
  > **classification methods**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 141 — SPLIT STILL NEEDED

*Your centers:* `research data, research result`

8 labels then, 5 of them still present (3 rejected in step 2), now in 1 cluster(s).

**→ new cluster `655`** (6 labels)

> **original_research_finding**, **research findings**, **Research outcome**, **research results**, **study findings**, type of research outcome

Command in step 4: inspect `655`, split, and paste:

```
research data, research result
```

*Rejected in step 2:* research data, research result, study data

---

## old cluster 122 — ALREADY SEPARATED

*Your centers:* `optimization method, optimization criteria`

8 labels then, 6 of them still present (2 rejected in step 2), now in 5 cluster(s).

The new clustering already pulled these apart:

- **cluster `659`** (1 labels) contains 1 of yours
  > **optimizer**
- **cluster `661`** (1 labels) contains 1 of yours
  > **optimization techniques**
- **cluster `662`** (1 labels) contains 1 of yours
  > **Optimization criteria**
- **cluster `663`** (1 labels) contains 1 of yours
  > **Optimization method**
- **cluster `1558`** (2 labels) contains 2 of yours
  > **has optimization**, **has optimization time**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* optimize, training optimization

---

## old cluster 113 — SPLIT STILL NEEDED

*Your centers:* `support material, supported by`

8 labels then, 8 of them still present (0 rejected in step 2), now in 1 cluster(s).

**→ new cluster `449`** (8 labels)

> **has support**, **has support URL**, **is supported by**, **statement supported by**, **Support material**, **Supported By**, **supporting data**, **Supports**

Command in step 4: inspect `449`, split, and paste:

```
support material, supported by
```

---

## old cluster 125 — SPLIT STILL NEEDED

*Your centers:* `recommendation, recommendation method`

8 labels then, 8 of them still present (0 rejected in step 2), now in 1 cluster(s).

**→ new cluster `338`** (9 labels)

> Future reccomendation, **has Recommended items**, **Recomm.**, **recommendation**, **Recommendation type**, **recommendations**, **recommender**, **uses Recommendation approach**, **uses Recommendation Method**

Command in step 4: inspect `338`, split, and paste:

```
recommendation, recommendation method
```

---

## old cluster 133 — ALREADY SEPARATED

*Your centers:* `system, system component`

8 labels then, 8 of them still present (0 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `258`** (11 labels) contains 2 of yours
  > Approach name, Framework / System Name, **Name of system**, Prototype System Name, System / Approach Name, System / Tool Name, **system name**, system_name, system_name_reference, Tool / System Name, Tool name
- **cluster `452`** (8 labels) contains 6 of yours
  > **System**, System Category, **System components**, **system feature**, **system information**, **System model**, **system module**, system_category

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 128 — ALREADY SEPARATED

*Your centers:* `metadata, metadata schema`

8 labels then, 8 of them still present (0 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `291`** (10 labels) contains 6 of yours
  > **abstract metadata**, **abstract-body metadata**, **abstract-heading metadata**, caption metadata, editor metadata, **introduction metadata**, **Metadata**, **other metadata**, text-body metadata, title metadata
- **cluster `1353`** (3 labels) contains 2 of yours
  > Metadata representation method, **Metadata schema**, **Metadata Standard Usage**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 109 — ALREADY SEPARATED

*Your centers:* `paper topic, paper type, paper template`

8 labels then, 7 of them still present (1 rejected in step 2), now in 5 cluster(s).

The new clustering already pulled these apart:

- **cluster `90`** (4 labels) contains 3 of yours
  > **has paper**, Paper, **paper class**, **Paper type**
- **cluster `92`** (2 labels) contains 1 of yours
  > **paper:type**, type of the paper
- **cluster `94`** (1 labels) contains 1 of yours
  > **paper template**
- **cluster `97`** (1 labels) contains 1 of yours
  > **paper_link**
- **cluster `98`** (1 labels) contains 1 of yours
  > **paper goal**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Paper Topic

---

## old cluster 119 — ALREADY SEPARATED

*Your centers:* `simulation, simulation approach, simulation results`

8 labels then, 8 of them still present (0 rejected in step 2), now in 6 cluster(s).

The new clustering already pulled these apart:

- **cluster `347`** (2 labels) contains 2 of yours
  > **Has simulation results**, **simulationResults**
- **cluster `348`** (2 labels) contains 2 of yours
  > **Simulation**, **simulation software**
- **cluster `349`** (1 labels) contains 1 of yours
  > **Simulator**
- **cluster `350`** (1 labels) contains 1 of yours
  > **simulates**
- **cluster `351`** (2 labels) contains 1 of yours
  > simulation method, **simulationMethodology**
- **cluster `352`** (1 labels) contains 1 of yours
  > **has simulation approach**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 171 — ALREADY SEPARATED

*Your centers:* `research application, research project, research target`

7 labels then, 6 of them still present (1 rejected in step 2), now in 5 cluster(s).

The new clustering already pulled these apart:

- **cluster `234`** (1 labels) contains 1 of yours
  > **research_target**
- **cluster `235`** (2 labels) contains 2 of yours
  > **Research_plan**, **research_project**
- **cluster `237`** (1 labels) contains 1 of yours
  > **research recommendation**
- **cluster `238`** (1 labels) contains 1 of yours
  > **research application**
- **cluster `241`** (1 labels) contains 1 of yours
  > **research direction**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* research_opportunity

---

## old cluster 191 — SPLIT STILL NEEDED

*Your centers:* `AI model, AI tool`

7 labels then, 7 of them still present (0 rejected in step 2), now in 1 cluster(s).

**→ new cluster `259`** (11 labels)

> **AI application**, AI approach, ai components, **AI Model**, **AI paradigm**, **AI techniques**, **AI technology**, **AI tool**, **ai_components**, Approach (ai), Core AI Approach

Command in step 4: inspect `259`, split, and paste:

```
AI model, AI tool
```

---

## old cluster 157 — SPLIT STILL NEEDED

*Your centers:* `cost, cost model`

7 labels then, 7 of them still present (0 rejected in step 2), now in 1 cluster(s).

**→ new cluster `468`** (7 labels)

> **Cost**, **Cost Effectiveness**, **Cost parameters**, **Has cost**, **Has cost factor**, **has cost model**, **Reported cost**

Command in step 4: inspect `468`, split, and paste:

```
cost, cost model
```

---

## old cluster 274 — GONE

*Your centers:* `Energy band gap, HOMO, LUMO`

5 labels then, 0 of them still present (5 rejected in step 2), now in 0 cluster(s).

None of these labels survived step 2 — nothing to do.

*Rejected in step 2:* Energy band gap, Energy band gap (eV), HOMO (eV), LUMO (eV), bandGap

---

## old cluster 264 — SPLIT STILL NEEDED

*Your centers:* `etch rate, frame rate, perception rate`

5 labels then, 4 of them still present (1 rejected in step 2), now in 1 cluster(s).

**→ new cluster `824`** (4 labels)

> **fps**, **frame rate**, **perception_rate**, **refresh_rate**

Command in step 4: inspect `824`, split, and paste:

```
etch rate, frame rate, perception rate
```

*Rejected in step 2:* etchRate

---

## old cluster 339 — ALREADY SEPARATED

*Your centers:* `number of training images, number of test images, number of validation images`

5 labels then, 2 of them still present (3 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `1384`** (3 labels) contains 1 of yours
  > # trained images, dataset contains images, **total numbers of images used for training**
- **cluster `1595`** (1 labels) contains 1 of yours
  > **Number of images**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Number of test images, Number of training images, Number of validation images

---

## old cluster 318 — ALREADY SEPARATED

*Your centers:* `number of sentences, number of test sentences, number of training sentences`

5 labels then, 3 of them still present (2 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `1233`** (2 labels) contains 2 of yours
  > **Number of test sentences**, **Number of training sentences**
- **cluster `1234`** (1 labels) contains 1 of yours
  > **Number of sentences**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* Number of development sentences, dataset size (sentences)

---

## old cluster 301 — ALREADY SEPARATED

*Your centers:* `training method, training time`

5 labels then, 4 of them still present (1 rejected in step 2), now in 3 cluster(s).

The new clustering already pulled these apart:

- **cluster `753`** (2 labels) contains 2 of yours
  > **training time**, **training time, s**
- **cluster `755`** (1 labels) contains 1 of yours
  > **training method**
- **cluster `756`** (1 labels) contains 1 of yours
  > **training**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* train

---

## old cluster 452 — SPLIT STILL NEEDED

*Your centers:* `false negative, false positive`

4 labels then, 2 of them still present (2 rejected in step 2), now in 1 cluster(s).

**→ new cluster `443`** (2 labels)

> **False Negative Rate**, **False Positive Rate**

Command in step 4: inspect `443`, split, and paste:

```
false negative, false positive
```

*Rejected in step 2:* False Negative, False Positive

---

## old cluster 473 — SPLIT STILL NEEDED

*Your centers:* `number of tokens, number of test tokens, number of training tokens`

4 labels then, 1 of them still present (3 rejected in step 2), now in 1 cluster(s).

**→ new cluster `3075`** (1 labels)

> **Number of tokens**

Command in step 4: inspect `3075`, split, and paste:

```
number of tokens, number of test tokens, number of training tokens
```

*Rejected in step 2:* Number of development tokens, Number of test tokens, Number of training tokens

---

## old cluster 387 — ALREADY SEPARATED

*Your centers:* `dependent variable, independent variable`

4 labels then, 3 of them still present (1 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `1060`** (2 labels) contains 2 of yours
  > **has dependent variables**, **Has independent variable**
- **cluster `1061`** (1 labels) contains 1 of yours
  > **Number of dependent variables**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

*Rejected in step 2:* has study design dependent variable

---

## old cluster 541 — GONE

*Your centers:* `class, order`

4 labels then, 0 of them still present (4 rejected in step 2), now in 0 cluster(s).

None of these labels survived step 2 — nothing to do.

*Rejected in step 2:* Class (Biology), Class (Taxonomy), Order (Taxonomy), Taxonomic group (Taxonomy)

---

## old cluster 720 — SPLIT STILL NEEDED

*Your centers:* `next step, previous step`

3 labels then, 3 of them still present (0 rejected in step 2), now in 1 cluster(s).

**→ new cluster `1161`** (3 labels)

> **has next step**, **has previois step**, **has previous step**

Command in step 4: inspect `1161`, split, and paste:

```
next step, previous step
```

---

## old cluster 670 — ALREADY SEPARATED

*Your centers:* `endogenous predictors, exogenous predictors`

3 labels then, 3 of them still present (0 rejected in step 2), now in 2 cluster(s).

The new clustering already pulled these apart:

- **cluster `1086`** (2 labels) contains 2 of yours
  > **endogenous predictors**, **exogenous predictors**
- **cluster `1087`** (1 labels) contains 1 of yours
  > **has endogenous variables**

Check whether this grouping matches your intent; if not, split the cluster(s) as before.

---

## old cluster 810 — SPLIT STILL NEEDED

*Your centers:* `ground truth answer, ground truth specification`

3 labels then, 2 of them still present (1 rejected in step 2), now in 1 cluster(s).

**→ new cluster `1033`** (3 labels)

> **ground truth provided**, Ground Truth Source, **Groundtruth Speci cation**

Command in step 4: inspect `1033`, split, and paste:

```
ground truth answer, ground truth specification
```

*Rejected in step 2:* ground truth answer

---

## old cluster 1601 — SPLIT STILL NEEDED

*Your centers:* `average recallm best average recall`

2 labels then, 2 of them still present (0 rejected in step 2), now in 1 cluster(s).

**→ new cluster `2041`** (2 labels)

> **Average Recall**, **Best Average Recall**

Command in step 4: inspect `2041`, split, and paste:

```
average recallm best average recall
```

---
