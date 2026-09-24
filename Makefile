# Talk To My Train — InnoTrans 2026 hackathon
# `make` or `make help` lists every command.

# ---- configuration (override on the command line, e.g. `make run PORT=9000`) ----
VENV       := .venv
PY         := $(VENV)/bin/python
DATA_DIR   := $(CURDIR)/data
IMAGE_NAME := ubahn-flow-dashboard
# Preferred dashboard port; the next free one is used automatically if it is taken.
PORT       ?= 8501
# Port of the ADK dev UI (make agent-web).
ADK_PORT   ?= 8000
# Small model for evaluation runs (the shared main model is never used by these targets unless you unset it).
EVAL_MODEL ?= gpt-4o-mini

# First free TCP port >= PORT. Pure Python, so it works on Linux and macOS with no extra tools.
FREE_PORT   = $(shell $(if $(wildcard $(PY)),$(PY),python3) -c "import socket,sys; p=int(sys.argv[1]); print(next(q for q in range(p, p+200) if socket.socket().connect_ex(('127.0.0.1', q))))" $(PORT))

.DEFAULT_GOAL := help
.PHONY: help up down status logs resources cognee-graph neo4j-up neo4j-down neo4j-sync adk-evalset venv install install-ml install-mcp install-agent install-all \
        run build docker-run docker-stop docker-logs \
        train-overcrowding train-disruption train-all checkpoints validate-pressure \
        mcp-server mcp-knowledge agent-query agent-cli agent-web demo \
        kb-build kb-sync kb-stats kg-seed kg-stats kg-export schemas \
        test check guardrail-suite eval-router eval ls-eval bench experiments experiments-report \
        backup-obs clean-obs clean clean-venv

##@ Help
help:  ## Show this list
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<command>\033[0m\n"} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nTypical first run:  make install-all  &&  make run\n\n"

##@ All services in one command
up:  ## Start everything in the background: dashboard, ADK chat UI, knowledge MCP server, Neo4j (prepares the knowledge base + graph first). WITH_DATA_MCP=1 also starts the data MCP server; NO_ADK=1 skips the ADK UI; NO_NEO4J=1 skips Neo4j
	@test -x $(PY) || { echo "No $(VENV) found — run 'make install-all' first."; exit 1; }
	@PORT=$(PORT) ADK_PORT=$(ADK_PORT) $(PY) scripts/services.py up $(if $(WITH_DATA_MCP),--with-data-mcp) $(if $(NO_ADK),--no-adk) $(if $(NO_NEO4J),--no-neo4j)

down:  ## Stop everything that `make up` started
	@$(PY) scripts/services.py down

status:  ## Show the running services, their ports and health
	@$(PY) scripts/services.py status

logs:  ## Follow the service logs (all, or one: make logs S=dashboard)
	@$(PY) scripts/services.py logs $(S)

resources:  ## Report every resource and whether it works: Cognee (health, datasets, graph, sessions), knowledge base, knowledge graph (SQLite + Neo4j), MCP tools, models, keys
	@$(PY) agent/resources.py 2>&1 | grep -v WARNING

cognee-graph:  ## Save Cognee's interactive knowledge-graph page (the memory) as .run/cognee_graph.html — the tenant page needs the API key, so it is kept as a local file
	@$(PY) agent/resources.py graph 2>&1 | grep -v WARNING

neo4j-up:  ## Create/start the local Neo4j (Docker image neo4j:5.26-community, data volume kept) and load the knowledge graph into it; needs NEO4J_PASSWORD in .env
	@grep -q '^NEO4J_PASSWORD=' .env || { echo "Add NEO4J_URI=bolt://localhost:7687, NEO4J_USER=neo4j, NEO4J_PASSWORD=<pw> to .env first."; exit 1; }
	@docker start nextmove-neo4j >/dev/null 2>&1 || docker run -d --name nextmove-neo4j --restart unless-stopped -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
		-v nextmove_neo4j_data:/data -e NEO4J_AUTH=neo4j/$$(grep '^NEO4J_PASSWORD=' .env | cut -d= -f2) neo4j:5.26-community >/dev/null
	@echo "waiting for Neo4j ..."; for i in $$(seq 1 40); do (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1 && break; sleep 2; done; sleep 6
	@$(PY) agent/kgraph.py neo4j-sync 2>&1 | grep -v WARNING | head -8; echo "Neo4j browser -> http://localhost:7474 (user neo4j)"

neo4j-sync:  ## Bring Neo4j level with the local knowledge graph (idempotent); ARGS=--clear empties Neo4j first
	$(PY) agent/kgraph.py neo4j-sync $(ARGS)

neo4j-down:  ## Stop the Neo4j container (the data volume is kept)
	-docker stop nextmove-neo4j

##@ Setup (all installs go into ./.venv)
venv:  ## Create the local .venv (Python 3) and upgrade pip
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: venv  ## Install the dashboard dependencies (Streamlit, Plotly, ...)
	$(PY) -m pip install -r dashboard/requirements.txt

install-ml: venv  ## Install the ML dependencies (TabPFN client, scikit-learn, pytest)
	$(PY) -m pip install -r ml/requirements.txt

install-mcp: venv  ## Install the MCP server dependencies (FastMCP)
	$(PY) -m pip install -r mcp_server/requirements.txt

install-agent: venv  ## Install the agent dependencies (Google ADK, LiteLLM, pydantic, LangSmith / openevals, LangChain graph extraction)
	$(PY) -m pip install -r agent/requirements.txt

install-all: install install-ml install-mcp install-agent  ## Install everything (dashboard + ML + MCP + agent)

##@ Dashboard
run:  ## Run the Streamlit dashboard locally (default port 8501, or the next free one)
	@test -x $(PY) || { echo "No $(VENV) found — run 'make install' first."; exit 1; }
	@port=$(FREE_PORT); \
	 [ "$$port" = "$(PORT)" ] || echo "Port $(PORT) is in use — using $$port instead."; \
	 echo "Dashboard → http://localhost:$$port"; \
	 cd dashboard && DATA_DIR="$(DATA_DIR)" ../$(PY) -m streamlit run app.py --server.port $$port

build:  ## Build the dashboard Docker image
	docker build -t $(IMAGE_NAME) dashboard

docker-run: build  ## Run the dashboard in Docker (mounts data/, ml/output/, observability/) on a free port
	@port=$(FREE_PORT); \
	 docker run --rm -d --name $(IMAGE_NAME) -p $$port:8501 \
		-v "$(DATA_DIR):/app/data:ro" \
		-v "$(CURDIR)/ml/output:/app/ml_output:ro" \
		-v "$(CURDIR)/observability:/app/observability" \
		$(IMAGE_NAME) && echo "Dashboard → http://localhost:$$port"

docker-stop:  ## Stop and remove the dashboard container
	-docker stop $(IMAGE_NAME)

docker-logs:  ## Follow the dashboard container logs
	docker logs -f $(IMAGE_NAME)

##@ ML engine (TabPFN — needs TABPFN_API_TOKEN in a .env file)
train-overcrowding:  ## Train + evaluate the overcrowding classifier -> ml/output/overcrowding_predictions.csv
	$(PY) ml/train_overcrowding_classifier.py

train-disruption:  ## Train + evaluate the flow regressor (Category C) and run the 26-closure case study (~3 min)
	$(PY) ml/train_disruption_baseline.py

train-all: train-overcrowding train-disruption  ## Regenerate every result file the dashboard's ML Engine page reads

checkpoints:  ## Save/verify the TabPFN model checkpoints used for inference (ml/checkpoints/); FORCE=1 refits all
	$(PY) ml/save_checkpoints.py $(if $(FORCE),--force)

validate-pressure:  ## Skill of the pressure ranking on replay days -> knowledge/validation.json (~2 min, TabPFN API)
	$(PY) evaluation/validate_pressure.py

##@ Agent & MCP servers
agent-query:  ## One-shot question through the agent, e.g. make agent-query Q="Line U9 suspended..."
	@test -n "$(Q)" || { echo 'Usage: make agent-query Q="your question"'; exit 1; }
	$(PY) agent/run_query.py "$(Q)"

agent-cli:  ## Chat with the agent in the terminal
	$(VENV)/bin/adk run agent/

agent-web:  ## Open the ADK dev UI (tool calls, traces) on port 8000
	$(VENV)/bin/adk web --port $(ADK_PORT) agent/

adk-evalset:  ## Write the 3 approximate-answer eval cases into the ADK UI eval set eval_set_1 (docs/test_questions.md, "ADK eval set")
	$(PY) evaluation/make_adk_evalset.py

demo:  ## Scripted 6-turn conversation (bounce, follow-up, explanation, history hit, decline); small models only
	EVALUATOR_LITELLM_MODEL=$(EVAL_MODEL) $(PY) evaluation/conversation_demo.py

mcp-server:  ## Run the data + analytics + TabPFN MCP server standalone over stdio
	$(PY) mcp_server/server.py

mcp-knowledge:  ## Run the knowledge / sanity-check / knowledge-graph MCP server (stdio; MCP_TRANSPORT=http MCP_PORT=8766 for HTTP)
	$(PY) mcp_server/knowledge_server.py

##@ Knowledge base, memory & knowledge graph
kb-build:  ## Build the knowledge base (ground truth / boundaries / insights) from the raw CSVs -> knowledge/knowledge.{json,md}
	$(PY) agent/knowledge_build.py

kb-sync:  ## Push the knowledge base to Cognee (graph build runs server-side; sends the curated knowledge to your Cognee tenant)
	$(PY) agent/knowledge.py sync

kb-stats:  ## Knowledge-base size, stored turns/sessions and Cognee connectivity
	$(PY) agent/knowledge.py stats

kg-seed:  ## Seed the local knowledge graph (26 closures, accepted answers, question bank, LLM extraction); ARGS=--no-llm skips the LLM step
	$(PY) agent/kgraph_build.py $(ARGS)

kg-stats:  ## Knowledge-graph size, node/relationship counts, top actions
	$(PY) agent/kgraph.py stats

kg-export:  ## Export the graph as Cypher for Neo4j / the LLM Graph Builder -> knowledge/kg_export.cypher
	$(PY) agent/kgraph.py export

schemas:  ## Write the JSON Schemas of every agent hand-over message -> docs/schemas/
	$(PY) agent/schemas.py

##@ Tests & evaluation (evaluation targets use the small model; the shared LLM endpoint is not hammered)
test:  ## Run the offline unit tests (no LLM, no TabPFN, no network)
	$(PY) -m pytest -q tests

guardrail-suite:  ## Scope / guardrail / routing / follow-up decisions on a labelled set (no LLM, no network, ~1 s)
	$(PY) evaluation/guardrail_suite.py

eval-router:  ## Accuracy of the deterministic question router on docs/test_questions.md (no LLM, no network)
	$(PY) agent/eval_router.py

check: test guardrail-suite eval-router  ## Everything that needs no LLM and no network: unit tests + guardrail suite + router accuracy

eval:  ## Score the agent (LLM judge + deterministic gates). Default: ONE brutal multi-part question. ARGS="--suite challenge --ids CH1,CH3", "--suite training --allow-many", "--cheap" = small writer
	EVALUATOR_LITELLM_MODEL=$(EVAL_MODEL) $(PY) evaluation/run_eval.py $(ARGS)

ls-eval:  ## LangSmith-style evaluation (openevals LLM judges + run metrics) of the latest stored runs, offline; ARGS="--upload" sends to LangSmith
	$(PY) evaluation/langsmith_eval.py $(ARGS)

bench:  ## Latency benchmark (8 questions, warm server); add ARGS="--show" to print the answers
	$(PY) agent/bench.py --wait $(ARGS)

experiments:  ## Component study (2 questions x 12 arms: router / memory / MCP transport / ML engine / writer); ARGS="--list" shows the design, "--arms A00,M1" a subset
	$(PY) experiments/run_experiments.py $(ARGS)

experiments-report:  ## Regenerate the results tables in docs/experiments_plan.md from the latest experiment run (or EXP=exp-...)
	$(PY) experiments/report.py $(EXP)

##@ Housekeeping
backup-obs:  ## Copy the observability database to observability/backup/ (do this before big evaluation runs)
	@mkdir -p observability/backup
	$(PY) -c "import sqlite3,time,sys; s=sqlite3.connect('observability/agent_obs.db'); d=sqlite3.connect('observability/backup/agent_obs_'+time.strftime('%Y%m%d_%H%M%S')+'.db'); s.backup(d); print('backed up ->', d.execute('pragma database_list').fetchone()[2])"

clean-obs:  ## Delete the observability database (runs, traces, eval results) — see backup-obs first
	rm -f observability/agent_obs.db observability/agent_obs.db-wal observability/agent_obs.db-shm

clean:  ## Delete all __pycache__ folders
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +

clean-venv:  ## Delete the .venv (you will need 'make install-all' again)
	rm -rf $(VENV)
