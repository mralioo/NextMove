# Talk To My Train — InnoTrans 2026 hackathon.  Four commands; `make` alone shows them.
#
#   make install   set up .venv and install every dependency (dashboard, ML, MCP, agent, evaluation)
#   make up        run everything: prepares the knowledge base + graph, then starts the dashboard, the ADK chat/eval UI,
#                  the knowledge MCP server and Neo4j; prints every address (options: WITH_DATA_MCP=1, NO_ADK=1, NO_NEO4J=1, PORT=, ADK_PORT=)
#   make down      stop everything `make up` started
#   make check     everything that needs no LLM and no network: unit tests + guardrail suite + router accuracy
#
# Everything else (status, logs, eval, resources, kb-*, kg-*, neo4j-*, ls-*, experiments, ML training, ...):
#   ./.venv/bin/python scripts/tasks.py            # lists every task with its description
#   ./.venv/bin/python scripts/tasks.py <task> ... # runs one (docs/running_the_system.md)

VENV := .venv
PY   := $(VENV)/bin/python
PORT     ?= 8501
ADK_PORT ?= 8000

.DEFAULT_GOAL := help
.PHONY: help install up down check

help:
	@printf '\n\033[1mTalk To My Train\033[0m\n\n'
	@printf '  \033[36mmake install\033[0m   Set up .venv and install every dependency\n'
	@printf '  \033[36mmake up\033[0m        Run everything: dashboard, ADK chat/eval UI, knowledge MCP server, Neo4j (prepares the knowledge base + graph first)\n'
	@printf '  \033[36mmake down\033[0m      Stop everything that `make up` started\n'
	@printf '  \033[36mmake check\033[0m     Unit tests + guardrail suite + router accuracy (no LLM, no network)\n\n'
	@printf '  All other tasks: ./.venv/bin/python scripts/tasks.py   (lists them; docs/running_the_system.md)\n\n'

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r dashboard/requirements.txt -r ml/requirements.txt -r mcp_server/requirements.txt -r agent/requirements.txt
	@command -v npm >/dev/null && (cd frontend && npm install --no-audit --no-fund && npm run build) || echo 'npm not found: skipping the React operator desktop (frontend/)'

up:
	@test -x $(PY) || { echo "No $(VENV) found — run 'make install' first."; exit 1; }
	@PORT=$(PORT) ADK_PORT=$(ADK_PORT) $(PY) scripts/services.py up $(if $(WITH_DATA_MCP),--with-data-mcp) $(if $(NO_ADK),--no-adk) $(if $(NO_NEO4J),--no-neo4j)

down:
	@$(PY) scripts/services.py down

check:
	$(PY) -m pytest -q tests
	$(PY) evaluation/guardrail_suite.py
	$(PY) agent/eval_router.py | head -1
