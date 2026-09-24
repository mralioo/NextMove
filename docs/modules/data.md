# `data/`, `knowledge/`, `observability/` — data and stores

Full schemas: [`../data_schema.md`](../data_schema.md). Raw files: [`../../data/dataset_schema.md`](../../data/dataset_schema.md). Golden files: [`../../data/data_schema_high_quality.md`](../../data/data_schema_high_quality.md).

| Folder | Content | Built by | Read by |
| --- | --- | --- | --- |
| `data/training dataset/` | organisers' files, 2026-06-10 → 2026-09-21 (`*_pre_innotrans.csv`, stations, connections, lines) | given | everything (merged with the testing split) |
| `data/testing dataset/` | organisers' held-out days 2026-09-22 → 2026-10-01 (`*_rest.csv`) | given | everything |
| `data/normalized/` | normal flow, normalized flows / weather / rest (train + `_test`), episodes, coefficients, model | `ml/nextmove_pipeline` (`tasks.py quality-build` / `quality-pipeline`) | `ml/golden_data.py`, `golden_*` tools, quality build |
| `data/processed/` | geocode cache | pipeline | pipeline, `golden_geocode_cache` |
| `data/quality/` | `quality.db` + `cells.npz`: boundaries, ceilings, episodes, anomalies, outages, data issues | `ml/quality_db.py build` | Inspector (`quality_check_facts`), `quality_*` tools |
| `knowledge/` | `knowledge.json` / `.md` (55 entries: 14 boundaries, 37 ground-truth facts, 4 insights), `validation.json`, `kg_export.cypher` | `agent/knowledge_build.py` (`kb-build`), `kg-export`, `validate-pressure` | Inspector, sanity check, knowledge MCP server |
| `observability/` | `memory.db` (turns, feedback, artifacts), `kgraph.db` (knowledge graph), `agent_obs.db` (runs, spans, evaluation, experiments), `submissions.db`, `backup/` | the running system | UI / API, dashboard, evaluation |
| `ml/cache/`, `ml/checkpoints/`, `ml/output/` | feature-table parquet, TabPFN checkpoints (`manifest.json`), model reports | `ml/*` | MCP data server |

## Rules

* **Never edit raw or golden files by hand.** Everything derived is rebuilt by a task (`make up` builds what is missing).
* The knowledge base is computed **from the raw CSVs**, not from the agent's tools.
* `observability/*.db` are local runtime state (git-ignored, not in the repository); `tasks.py backup-obs` copies the observability database; `clean-obs` deletes it.
* `.env` and keys are never committed.

## Caveats (also in the golden schema)

Flows are simulated (the flow files have 168 station columns, one of them a duplicate interchange column `U Stadtmitte (Berlin).1`, i.e. 167 distinct stations); no U4 line data; the energy file is daily per line; the events file mixes real events with an estimated attendance; closures are simulated; dates outside 2026-06-10 → 2026-10-01 can only be answered as clearly labelled scenarios.
