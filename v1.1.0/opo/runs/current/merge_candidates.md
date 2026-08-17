# Merge candidates for the step 4 review

Cluster pairs whose centroids are closest in the embedding space — the places where
step 3 most likely split one concept into several clusters. Ordered by similarity.

In the review type `merge <a> <b>` to fold the second cluster into the first.

| similarity | command | cluster A | cluster B |
| --- | --- | --- | --- |
| 0.923 | `merge 24 25` | Evaluator, evaluation, evaluation aspect, evaluation item | Evaluation methods, Means of evaluation, evaluation measures, evaluation method |
| 0.92 | `merge 223 373` | # of languages, Data Language, Focus languages, covered languages, dataset language, focus_language … | Language, Language Genus, Language/domain, Languages, Subject language, language domain … |
| 0.919 | `merge 72 73` | Meaured in, is measured in, measured, measured by, measured for, measured_by … | measure, measurement, measurement value, measurementMethod, measurementTechnique |
| 0.913 | `merge 1 4` | Ontologies used, Ontology used, Used Ontologies, Used Upper Ontologies, Uses ontology, uses_ontolology … | Ontologies, Ontology, Ontology domains |
| 0.913 | `merge 25 26` | Evaluation methods, Means of evaluation, evaluation measures, evaluation method | Evaluation metics, Evaluation metric, Evaluation metrics, Has evaluation metrics, evaluation_metric, evaluation_metrics |
| 0.911 | `merge 91 93` | paper: publication_year, paper:publicationDate, paper:publication_year, paper:puplication_year, paper:year, paper_year | paper:journal, paper:publicationTypes, paper:publised_in, paper:published_in, paper:venue |
| 0.908 | `merge 4 10` | Ontologies, Ontology, Ontology domains | om ontology, prov ontology, qu ontology |
| 0.908 | `merge 320 321` | score calculation via, scoring method | score, score details, score type, score value, scores |
| 0.908 | `merge 77 78` | Time duration, Time preiod, time, time interval, timestamp, timing | Time frame, has time, has time frame |
| 0.908 | `merge 10 11` | om ontology, prov ontology, qu ontology | ontology based, ontology component, ontology module |
| 0.906 | `merge 27 28` | evaluated against, evaluates | Evaluated on, evaluate on, evaluate-on |
| 0.904 | `merge 99 103` | existing datasets utilization, used datasets, utilized datasets | Dataset used, Datasets used |
| 0.903 | `merge 157 184` |  Cites / Source, # Sources, Claim Sources, Information sources, Other sources/ (papers), Source domain … | Data origin, Data source content, content_source, data resources, data source, data source description … |
| 0.903 | `merge 488 490` | sentence error rate, sentence error rate (SER), word error rate | character error rate, character error rate (CER) |
| 0.903 | `merge 4 11` | Ontologies, Ontology, Ontology domains | ontology based, ontology component, ontology module |
| 0.901 | `merge 25 44` | Evaluation methods, Means of evaluation, evaluation measures, evaluation method | evaluation criteria, evaluation_criteria_selection |
| 0.9 | `merge 475 476` | hreshold measurement, threshold measurement, threshold method, threshold setting | activation_threshold, trigger threshold |
| 0.899 | `merge 3 10` | Ontology name, Ontology type | om ontology, prov ontology, qu ontology |
| 0.899 | `merge 3 4` | Ontology name, Ontology type | Ontologies, Ontology, Ontology domains |
| 0.899 | `merge 354 805` | assessment, assessment test | Test, Tests, test form, test score |
| 0.899 | `merge 186 187` | Year, Year (end), Year data | Year (start), start year |
| 0.898 | `merge 24 27` | Evaluator, evaluation, evaluation aspect, evaluation item | evaluated against, evaluates |
| 0.897 | `merge 435 437` | Data property count, Number of properties, properties count | Object property count, no of object properties, number of object properties |
| 0.897 | `merge 1 10` | Ontologies used, Ontology used, Used Ontologies, Used Upper Ontologies, Uses ontology, uses_ontolology … | om ontology, prov ontology, qu ontology |
| 0.896 | `merge 24 28` | Evaluator, evaluation, evaluation aspect, evaluation item | Evaluated on, evaluate on, evaluate-on |
| 0.896 | `merge 72 74` | Meaured in, is measured in, measured, measured by, measured for, measured_by … | measured during, measured_during |
| 0.895 | `merge 25 33` | Evaluation methods, Means of evaluation, evaluation measures, evaluation method | Evaluation Benchmarks, evaluation benchmark |
| 0.895 | `merge 164 166` | ACM Digital Library author ID, DBLP author ID, Google Scholar ID, OpenAlex ID, Scopus author ID, Semantic Scholar author ID | ResearchGate ID, ResearcherID |
| 0.895 | `merge 1 23` | Ontologies used, Ontology used, Used Ontologies, Used Upper Ontologies, Uses ontology, uses_ontolology … | Reused ontology, reused ontologies |
| 0.895 | `merge 283 284` | interactionType, interactivityType | Type of interaction, interaction_type, interaction_types |
| 0.894 | `merge 27 353` | evaluated against, evaluates | assessed, assessed by, assessed_by, assesses |
| 0.893 | `merge 24 44` | Evaluator, evaluation, evaluation aspect, evaluation item | evaluation criteria, evaluation_criteria_selection |
| 0.893 | `merge 429 431` | average recall CEA, average recall CPA, average recall CTA | average precision CEA, average precision CPA, average precision CTA |
| 0.893 | `merge 68 73` | measured physical quantity, variableMeasured | measure, measurement, measurement value, measurementMethod, measurementTechnique |
| 0.893 | `merge 138 140` | Published, publication, published by, published in, published_in | Has published, has published in |
| 0.893 | `merge 46 47` | Experiment name, experiment id | experiment, experiment on, experiment type, experiment with, type of experiment |
| 0.892 | `merge 402 404` | Application architecture, software_architecture | architecture type, system architecture |
| 0.891 | `merge 78 82` | Time frame, has time, has time frame | At time period, time period |
| 0.89 | `merge 3 11` | Ontology name, Ontology type | ontology based, ontology component, ontology module |
| 0.89 | `merge 68 72` | measured physical quantity, variableMeasured | Meaured in, is measured in, measured, measured by, measured for, measured_by … |
| 0.889 | `merge 111 287` | App. Type, Application Domain, Application area, Application field, Application field(s), Application type … | Applicable condition, Has application in, applicability, applicable in, applied on, applied to … |
| 0.889 | `merge 77 82` | Time duration, Time preiod, time, time interval, timestamp, timing | At time period, time period |
| 0.888 | `merge 157 218` |  Cites / Source, # Sources, Claim Sources, Information sources, Other sources/ (papers), Source domain … | Knowledge source, Type of knowledge source, knowledge_source, knowledge_source_reference |
| 0.888 | `merge 267 268` | trained for, trained-on, transformer trained on | Pretrained model, pre-trained for, pre-trained model used, pre-trained on |
| 0.888 | `merge 47 48` | experiment, experiment on, experiment type, experiment with, type of experiment | Experimental details, experiment description, experiment details |
| 0.888 | `merge 324 326` | accessibilityAPI, accessibilityControl, accessibilityFeature, accessibility_features | accessibilitySummary |
| 0.888 | `merge 73 75` | measure, measurement, measurement value, measurementMethod, measurementTechnique | Measurement type |
| 0.888 | `merge 136 502` |  HAS_RESULTS, Has res, Has_result, QA results, Results from, analysis result … | Outcome, Outcome indicator, Outcome of interaction, has bounded outcome, has outcome, interaction_outcome … |
| 0.887 | `merge 427 428` | Retrieval Mechanism, Retrieval Strategy, Retrieval-based, retrieval process, retrieval_mechanism, retrieval_mechanism_reference | retrieval source, retrieved |
| 0.887 | `merge 391 393` | tool  description, tool URL, tool useses AI | Tool, Tools |
| 0.886 | `merge 260 263` | AUC-ROC, AUROC, ROC area, Receiver Operating Characteristic (ROC) area, receiver operating characteristic, roc auc | AUCPR, Precision-Recall Curve (PRC) area |
| 0.886 | `merge 90 98` | Paper, Paper type, has paper, paper class | paper goal |
| 0.886 | `merge 399 469` | 07- Model 2, model, model family, model framework, model name, model type … | backbone, backbone model, background model, base model, base_model, contribution:base_model … |
| 0.885 | `merge 511 512` | NLP data format, NLP dataset | NLP data source, NLP data source domain |
| 0.885 | `merge 114 116` | 1st step, 2nd step, Step 1, Step 2, Step 2.1, Step 2.1.1 | 3rd step, Step 3, Step 4 |
| 0.885 | `merge 0 4` | Schema ontology, ontology design pattern | Ontologies, Ontology, Ontology domains |
| 0.884 | `merge 122 123` | observation data, observation number, observation type, observation type description, observation_data | Observation, is direct observation |
| 0.884 | `merge 33 34` | Evaluation Benchmarks, evaluation benchmark | evaluation_framework, evaluation_frameworks |
| 0.884 | `merge 301 303` | # participants, Number of participants, Total participants, amount of participants, has number of participants, number of attending experts … | Has participant, Has participants, Has participating person, Type of participant, category of participant, has participants characteristic … |
| 0.884 | `merge 138 168` | Published, publication, published by, published in, published_in | PublicationDate, PublicationYear, Publish year, Publishing Date, Year of publication, copyright year … |
