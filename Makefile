IMAGE_NAME := ubahn-flow-dashboard
CONTAINER_NAME := ubahn-flow-dashboard
PORT := 8501
ADK_PORT := 8000
DATA_DIR := $(CURDIR)/data

.PHONY: help venv install run build docker-run docker-stop docker-logs clean install-ml train-overcrowding install-mcp mcp-server install-agent agent-query agent-web agent-cli

help:
	@echo "Targets:"
	@echo "  venv               Create local .venv"
	@echo "  install            Install dashboard/requirements.txt into .venv"
	@echo "  run                Run the Streamlit dashboard locally (no Docker)"
	@echo "  build              Build the Docker image"
	@echo "  docker-run         Run the dashboard in Docker (mounts ./data), open http://localhost:$(PORT)"
	@echo "  docker-stop        Stop and remove the running container"
	@echo "  docker-logs        Follow the container logs"
	@echo "  install-ml         Install ml/requirements.txt into .venv (TabPFN client etc.)"
	@echo "  train-overcrowding Train the TabPFN overcrowding-risk classifier (needs TABPFN_API_TOKEN in .env)"
	@echo "  install-mcp        Install mcp_server/requirements.txt into .venv"
	@echo "  mcp-server         Run the MCP dataset/TabPFN server standalone (stdio)"
	@echo "  install-agent      Install agent/requirements.txt into .venv (Google ADK etc.)"
	@echo "  agent-query Q=...  Run one question through the ADK agent, plain text output (needs an LLM key)"
	@echo "  agent-web          Start the ADK dev UI to visualize the agent (tool calls, traces) at http://localhost:$(ADK_PORT)"
	@echo "  agent-cli          Start an interactive ADK terminal chat with the agent"
	@echo "  clean              Remove local .venv and __pycache__ files"

venv:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip

install: venv
	./.venv/bin/pip install -r dashboard/requirements.txt

run:
	DATA_DIR="$(DATA_DIR)" ./.venv/bin/streamlit run dashboard/app.py

build:
	docker build -t $(IMAGE_NAME) dashboard

docker-run: build
	docker run --rm -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8501 \
		-v "$(DATA_DIR):/app/data:ro" \
		$(IMAGE_NAME)
	@echo "Dashboard running at http://localhost:$(PORT)"

docker-stop:
	-docker stop $(CONTAINER_NAME)

docker-logs:
	docker logs -f $(CONTAINER_NAME)

install-ml: venv
	./.venv/bin/pip install -r ml/requirements.txt

train-overcrowding:
	./.venv/bin/python ml/train_overcrowding_classifier.py

install-mcp: venv
	./.venv/bin/pip install -r mcp_server/requirements.txt

mcp-server:
	./.venv/bin/python mcp_server/server.py

install-agent: venv
	./.venv/bin/pip install -r agent/requirements.txt

agent-query:
	./.venv/bin/python agent/run_query.py "$(Q)"

agent-web:
	./.venv/bin/adk web --port $(ADK_PORT) agent/

agent-cli:
	./.venv/bin/adk run agent/

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
