"""Everything that is not one of the four `make` commands (install / up / down / check).

    ./.venv/bin/python scripts/tasks.py                 # list every task with its description
    ./.venv/bin/python scripts/tasks.py <task> [args]   # run one; extra args are passed to the underlying command

Examples
    scripts/tasks.py eval --suite challenge --ids CH1,CH3 --cheap
    scripts/tasks.py agent-query "Line U9 is suspended between ... why?"
    scripts/tasks.py experiments-report exp-20260923-225649
    scripts/tasks.py logs dashboard
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python") if (REPO / ".venv" / "bin" / "python").exists() else sys.executable
ADK = str(REPO / ".venv" / "bin" / "adk")
SMALL = os.environ.get("EVAL_MODEL", "gpt-4o-mini")        # evaluation tasks use the small model; the shared main model is not touched

# name: (description, argv, extra environment)
TASKS: dict[str, tuple[str, list[str], dict]] = {
    # ---- running
    "status": ("Running services, ports and health", [PY, "scripts/services.py", "status"], {}),
    "logs": ("Follow the service logs (all, or: logs <service>)", [PY, "scripts/services.py", "logs"], {}),
    "agent-query": ('One-shot question through the agent: agent-query "your question"', [PY, "agent/run_query.py"], {}),
    "agent-cli": ("Chat with the agent in the terminal", [ADK, "run", "agent/"], {}),
    "demo": ("Scripted 6-turn conversation (bounce, follow-up, explanation, history hit, decline); small models", [PY, "evaluation/conversation_demo.py"], {"EVALUATOR_LITELLM_MODEL": SMALL}),
    "mcp-server": ("Data + analytics + TabPFN MCP server over stdio", [PY, "mcp_server/server.py"], {}),
    "mcp-knowledge": ("Knowledge / sanity-check / graph MCP server over stdio", [PY, "mcp_server/knowledge_server.py"], {}),
    # ---- resources
    "resources": ("Report every resource and whether it works (Cognee, KB, graph, Neo4j, MCP tools, models, keys) with links to their UIs", [PY, "agent/resources.py"], {}),
    "cognee-graph": ("Save Cognee's interactive memory graph as .run/cognee_graph.html", [PY, "agent/resources.py", "graph"], {}),
    "neo4j-up": ("Create/start the local Neo4j (Docker) and load the knowledge graph into it", [PY, "scripts/neo4j.py", "up"], {}),
    "neo4j-sync": ("Bring Neo4j level with the local knowledge graph (--clear empties Neo4j first)", [PY, "agent/kgraph.py", "neo4j-sync"], {}),
    "neo4j-down": ("Stop the Neo4j container (data volume kept)", [PY, "scripts/neo4j.py", "down"], {}),
    # ---- knowledge
    "kb-build": ("Build the knowledge base (ground truth / boundaries / insights) from the raw CSVs", [PY, "agent/knowledge_build.py"], {}),
    "kb-sync": ("Push the knowledge base to Cognee (sends the curated knowledge to your tenant)", [PY, "agent/knowledge.py", "sync"], {}),
    "kg-seed": ("Seed the local knowledge graph (26 closures, accepted answers, question bank, LLM extraction; --no-llm skips it)", [PY, "agent/kgraph_build.py"], {}),
    "kg-export": ("Export the graph as Cypher -> knowledge/kg_export.cypher", [PY, "agent/kgraph.py", "export"], {}),
    "schemas": ("Write the JSON Schemas of every agent hand-over message -> docs/schemas/", [PY, "agent/schemas.py"], {}),
    # ---- ML engine
    "checkpoints": ("Save/verify the TabPFN checkpoints (--force refits)", [PY, "ml/save_checkpoints.py"], {}),
    "train-overcrowding": ("Train + evaluate the overcrowding classifier", [PY, "ml/train_overcrowding_classifier.py"], {}),
    "train-disruption": ("Train + evaluate the flow regressor and run the 26-closure case study (~3 min)", [PY, "ml/train_disruption_baseline.py"], {}),
    "validate-pressure": ("Skill of the pressure ranking on replay days -> knowledge/validation.json (~2 min)", [PY, "evaluation/validate_pressure.py"], {}),
    # ---- evaluation
    "guardrail-suite": ("Scope / guardrail / routing / follow-up decisions on a labelled set (no LLM)", [PY, "evaluation/guardrail_suite.py"], {}),
    "eval-router": ("Accuracy of the deterministic router on docs/test_questions.md (no LLM)", [PY, "agent/eval_router.py"], {}),
    "eval": ("Score the agent (LLM judge + gates): default ONE brutal question; --suite challenge|training, --ids CH1,CH3, --cheap, --allow-many", [PY, "evaluation/run_eval.py"], {"EVALUATOR_LITELLM_MODEL": SMALL}),
    "adk-evalset": ("Write the 3 approximate-answer cases into the ADK UI eval set eval_set_1", [PY, "evaluation/make_adk_evalset.py"], {}),
    "ls-eval": ("LangSmith-style evaluation of the latest stored runs, offline (--upload sends to LangSmith)", [PY, "evaluation/langsmith_eval.py"], {}),
    "ls-status": ("LangSmith: is LANGSMITH_API_KEY set and valid", [PY, "evaluation/langsmith_run.py", "status"], {}),
    "ls-dataset": ("LangSmith: create/update the test dataset `nextmove-eval` (uploads 8 questions with reference answers)", [PY, "evaluation/langsmith_run.py", "dataset"], {}),
    "ls-run": ("LangSmith: run the agent on the core examples and upload the experiment (--all, --offline)", [PY, "evaluation/langsmith_run.py", "run"], {}),
    "experiments": ("Component study, 2 questions x 10 arms (--list shows the design, --arms A00,M1)", [PY, "experiments/run_experiments.py"], {}),
    "experiments-report": ("Regenerate the results tables in docs/experiments_plan.md (optionally an experiment id)", [PY, "experiments/report.py"], {}),
    # ---- housekeeping
    "backup-obs": ("Copy the observability database to observability/backup/", [PY, "scripts/neo4j.py", "backup-obs"], {}),
    "clean-obs": ("Delete the observability database (runs, traces, eval results) — back it up first", [PY, "scripts/neo4j.py", "clean-obs"], {}),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "list"):
        print(__doc__)
        w = max(map(len, TASKS))
        for name, (desc, _, _) in TASKS.items():
            print(f"  {name:{w}s}  {desc}")
        return
    name, extra = sys.argv[1], sys.argv[2:]
    if name not in TASKS:
        raise SystemExit(f"unknown task '{name}'. Run without arguments to list them.")
    _, argv, env = TASKS[name]
    raise SystemExit(subprocess.call(argv + extra, cwd=REPO, env={**os.environ, **env}))


if __name__ == "__main__":
    main()
