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

# First free TCP port >= PORT. Pure Python, so it works on Linux and macOS with no extra tools.
FREE_PORT   = $(shell $(if $(wildcard $(PY)),$(PY),python3) -c "import socket,sys; p=int(sys.argv[1]); print(next(q for q in range(p, p+200) if socket.socket().connect_ex(('127.0.0.1', q))))" $(PORT))

.DEFAULT_GOAL := help
.PHONY: help venv install install-ml install-mcp install-agent install-all \
        run build docker-run docker-stop docker-logs \
        train-overcrowding train-disruption train-all checkpoints test \
        mcp-server agent-query agent-web agent-cli bench eval-router clean clean-venv

##@ Help
help:  ## Show this list
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<command>\033[0m\n"} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nTypical first run:  make install-all  &&  make run\n\n"

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

install-agent: venv  ## Install the agent dependencies (Google ADK, LiteLLM)
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

docker-run: build  ## Run the dashboard in Docker (mounts data/ and ml/output/) on a free port
	@port=$(FREE_PORT); \
	 docker run --rm -d --name $(IMAGE_NAME) -p $$port:8501 \
		-v "$(DATA_DIR):/app/data:ro" \
		-v "$(CURDIR)/ml/output:/app/ml_output:ro" \
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

test:  ## Run the offline unit tests (graph/closure logic; no API calls)
	$(PY) -m pytest -q tests

##@ MCP server & agent
mcp-server:  ## Run the MCP server standalone over stdio (dataset + TabPFN + Category C tools)
	$(PY) mcp_server/server.py

agent-query:  ## One-shot question through the agent, e.g. make agent-query Q="Line U9 suspended..."
	@test -n "$(Q)" || { echo 'Usage: make agent-query Q="your question"'; exit 1; }
	$(PY) agent/run_query.py "$(Q)"

agent-web:  ## Open the ADK dev UI (tool calls, traces) on port 8000
	$(VENV)/bin/adk web --port $(ADK_PORT) agent/

bench:  ## Latency benchmark of the fast pipeline (8 questions, warm server); add ARGS="--show" to print answers
	$(PY) agent/bench.py --wait $(ARGS)

eval-router:  ## Accuracy of the deterministic question router on docs/test_questions.md (no LLM, no network)
	$(PY) agent/eval_router.py

agent-cli:  ## Chat with the agent in the terminal
	$(VENV)/bin/adk run agent/

##@ Housekeeping
clean:  ## Delete all __pycache__ folders
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +

clean-venv:  ## Delete the .venv (you will need 'make install-all' again)
	rm -rf $(VENV)
