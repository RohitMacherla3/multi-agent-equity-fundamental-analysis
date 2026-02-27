SHELL := /bin/bash

ROOT := /Users/rohitmacherla/Documents/Projects/equities-research-agent
JAVA17_HOME := $(shell /usr/libexec/java_home -v 17 2>/dev/null)
RUN_DIR := $(ROOT)/.run
LOG_DIR := $(ROOT)/data/logs
GRADLE_HOME_INGEST := /tmp/mefa-gradle/ingestion
GRADLE_HOME_ORCH := /tmp/mefa-gradle/orchestration

INGEST_DIR := $(ROOT)/services/ingestion-service
INDEX_DIR := $(ROOT)/services/indexing-service
AI_DIR := $(ROOT)/services/ai-research-team-service
ORCH_DIR := $(ROOT)/services/orchestration-service
UI_DIR := $(ROOT)/services/demo-ui

INGEST_PID := $(RUN_DIR)/ingestion-service.pid
INDEX_PID := $(RUN_DIR)/indexing-service.pid
AI_PID := $(RUN_DIR)/ai-research-team-service.pid
ORCH_PID := $(RUN_DIR)/orchestration-service.pid

INGEST_LOG := $(LOG_DIR)/java-ingestion-service.log
INDEX_LOG := $(LOG_DIR)/python-indexing-service.log
AI_LOG := $(LOG_DIR)/python-ai-service.log
ORCH_LOG := $(LOG_DIR)/java-orchestration-service.log

.PHONY: all-services stop-all restart-all status wait-services \
	start-ingestion stop-ingestion restart-ingestion status-ingestion \
	start-indexing stop-indexing restart-indexing status-indexing \
	start-ai stop-ai restart-ai status-ai \
	start-orchestration stop-orchestration restart-orchestration status-orchestration \
	demo-ui

all-services: start-ingestion start-ai start-indexing start-orchestration wait-services
	@echo "All backend services started and healthy."

stop-all: stop-orchestration stop-ai stop-indexing stop-ingestion
	@echo "All backend services stopped."

restart-all: stop-all all-services

status: status-ingestion status-indexing status-ai status-orchestration

wait-services:
	@echo "Waiting for services to become healthy..."
	@for i in $$(seq 1 300); do \
		OK_ING=0; OK_AI=0; OK_IDX=0; OK_ORCH=0; \
		curl -fsS http://localhost:8080/actuator/health >/dev/null 2>&1 && OK_ING=1 || true; \
		curl -fsS http://localhost:8000/health >/dev/null 2>&1 && OK_AI=1 || true; \
		curl -fsS http://localhost:8002/health >/dev/null 2>&1 && OK_IDX=1 || true; \
		curl -fsS http://localhost:8081/health >/dev/null 2>&1 && OK_ORCH=1 || true; \
		if [ $$((i % 10)) -eq 0 ]; then \
			echo "health-check: ingestion=$$OK_ING ai=$$OK_AI indexing=$$OK_IDX orchestration=$$OK_ORCH (t=$$((i*2))s)"; \
		fi; \
		if [ $$OK_ING -eq 1 ] && [ $$OK_AI -eq 1 ] && [ $$OK_IDX -eq 1 ] && [ $$OK_ORCH -eq 1 ]; then \
			echo "All services are healthy."; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "Timed out waiting for health. Run 'make status' and inspect logs in $(LOG_DIR)." ; \
	exit 1

$(RUN_DIR):
	@mkdir -p $(RUN_DIR)

$(LOG_DIR):
	@mkdir -p $(LOG_DIR)

start-ingestion: | $(RUN_DIR) $(LOG_DIR)
	@if lsof -ti tcp:8080 >/dev/null 2>&1; then \
		echo "ingestion-service already running on :8080"; \
	else \
		echo "Starting ingestion-service..."; \
		: > "$(INGEST_LOG)"; \
		nohup env \
			JAVA_HOME="$(JAVA17_HOME)" \
			GRADLE_USER_HOME="$(GRADLE_HOME_INGEST)" \
			INGESTION_H2_PATH="$(INGEST_DIR)/data/ifip_local" \
			INGESTION_RAW_STORAGE_PATH="$(INGEST_DIR)/data/raw-filings" \
			GRADLE_OPTS='-Dorg.gradle.native=false' \
			gradle -p "$(INGEST_DIR)" bootRun --args='--spring.profiles.active=local' \
			>> "$(INGEST_LOG)" 2>&1 & echo $$! > "$(INGEST_PID)"; \
		echo "ingestion-service pid=$$(cat "$(INGEST_PID)") log=$(INGEST_LOG)"; \
	fi

stop-ingestion:
	@if [ -f "$(INGEST_PID)" ]; then kill -9 "$$(cat "$(INGEST_PID)")" >/dev/null 2>&1 || true; rm -f "$(INGEST_PID)"; fi
	@lsof -ti tcp:8080 | xargs kill -9 2>/dev/null || true
	@echo "ingestion-service stopped"

restart-ingestion: stop-ingestion start-ingestion

status-ingestion:
	@if lsof -ti tcp:8080 >/dev/null 2>&1; then echo "ingestion-service: UP (:8080)"; else echo "ingestion-service: DOWN"; fi

start-indexing: | $(RUN_DIR) $(LOG_DIR)
	@if lsof -ti tcp:8002 >/dev/null 2>&1; then \
		echo "indexing-service already running on :8002"; \
	else \
		echo "Starting indexing-service..."; \
		: > "$(INDEX_LOG)"; \
		nohup "$(AI_DIR)/.venv/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port 8002 \
			--app-dir "$(INDEX_DIR)" >> "$(INDEX_LOG)" 2>&1 & echo $$! > "$(INDEX_PID)"; \
		echo "indexing-service pid=$$(cat "$(INDEX_PID)") log=$(INDEX_LOG)"; \
	fi

stop-indexing:
	@if [ -f "$(INDEX_PID)" ]; then kill -9 "$$(cat "$(INDEX_PID)")" >/dev/null 2>&1 || true; rm -f "$(INDEX_PID)"; fi
	@lsof -ti tcp:8002 | xargs kill -9 2>/dev/null || true
	@echo "indexing-service stopped"

restart-indexing: stop-indexing start-indexing

status-indexing:
	@if lsof -ti tcp:8002 >/dev/null 2>&1; then echo "indexing-service: UP (:8002)"; else echo "indexing-service: DOWN"; fi

start-ai: | $(RUN_DIR) $(LOG_DIR)
	@if lsof -ti tcp:8000 >/dev/null 2>&1; then \
		echo "ai-research-team-service already running on :8000"; \
	else \
		echo "Starting ai-research-team-service..."; \
		: > "$(AI_LOG)"; \
		nohup env \
			AGENT_ENGINE=langgraph \
			AGENT_SWARM_ENABLE_NEWS=true \
			NEWS_PROVIDER=tavily \
			"$(AI_DIR)/.venv/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
			--app-dir "$(AI_DIR)" >> "$(AI_LOG)" 2>&1 & echo $$! > "$(AI_PID)"; \
		echo "ai-research-team-service pid=$$(cat "$(AI_PID)") log=$(AI_LOG)"; \
	fi

stop-ai:
	@if [ -f "$(AI_PID)" ]; then kill -9 "$$(cat "$(AI_PID)")" >/dev/null 2>&1 || true; rm -f "$(AI_PID)"; fi
	@lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
	@echo "ai-research-team-service stopped"

restart-ai: stop-ai start-ai

status-ai:
	@if lsof -ti tcp:8000 >/dev/null 2>&1; then echo "ai-research-team-service: UP (:8000)"; else echo "ai-research-team-service: DOWN"; fi

start-orchestration: | $(RUN_DIR) $(LOG_DIR)
	@if lsof -ti tcp:8081 >/dev/null 2>&1; then \
		echo "orchestration-service already running on :8081"; \
	else \
		echo "Starting orchestration-service..."; \
		: > "$(ORCH_LOG)"; \
		nohup env \
			JAVA_HOME="$(JAVA17_HOME)" \
			GRADLE_USER_HOME="$(GRADLE_HOME_ORCH)" \
			GRADLE_OPTS='-Dorg.gradle.native=false' \
			INGESTION_BASE_URL=http://localhost:8080 \
			INDEXING_BASE_URL=http://localhost:8002 \
			AI_BASE_URL=http://localhost:8000 \
			gradle -p "$(ORCH_DIR)" bootRun >> "$(ORCH_LOG)" 2>&1 & echo $$! > "$(ORCH_PID)"; \
		echo "orchestration-service pid=$$(cat "$(ORCH_PID)") log=$(ORCH_LOG)"; \
	fi

stop-orchestration:
	@if [ -f "$(ORCH_PID)" ]; then kill -9 "$$(cat "$(ORCH_PID)")" >/dev/null 2>&1 || true; rm -f "$(ORCH_PID)"; fi
	@lsof -ti tcp:8081 | xargs kill -9 2>/dev/null || true
	@echo "orchestration-service stopped"

restart-orchestration: stop-orchestration start-orchestration

status-orchestration:
	@if lsof -ti tcp:8081 >/dev/null 2>&1; then echo "orchestration-service: UP (:8081)"; else echo "orchestration-service: DOWN"; fi

demo-ui:
	@"$(AI_DIR)/.venv/bin/streamlit" run "$(UI_DIR)/app.py" --server.port 8501 --server.headless true
