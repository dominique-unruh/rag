# rag — common tasks. Run `make help` for the list.

VENV           ?= .venv
PY             := $(VENV)/bin/python
UV             ?= uv
PYTHON_VERSION ?=         # empty: use uv's default interpreter (must be >=3.10).
                          # Override to pin, e.g. PYTHON_VERSION=3.13
STORAGE_DIR    ?= storage

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(PY):
	$(UV) venv --allow-existing $(if $(strip $(PYTHON_VERSION)),--python $(PYTHON_VERSION),) $(VENV)

# Always run: `uv pip install` is idempotent and near-instant once satisfied, so
# other targets depend on it and a fresh/partial venv self-heals. (A cached stamp
# can go stale when the venv is recreated but the stamp file survives.)
.PHONY: install
install: $(PY) ## Create the venv and install deps (incl. test extra)
	$(UV) pip install --quiet --python $(PY) -e '.[test]'

.PHONY: test
test: install ## Run the unit test suite
	$(PY) -m pytest

PIPE_ARGS ?= --verbose

.PHONY: extract
extract: install ## Pipeline 1: sources -> Document Store
	$(PY) -m rag.extract $(PIPE_ARGS)

.PHONY: embed
embed: install ## Pipeline 2: Document Store -> Qdrant vectors
	$(PY) -m rag.embed $(PIPE_ARGS)

.PHONY: graph
graph: install ## Pipeline 3: Document Store -> Kùzu property graph
	$(PY) -m rag.graph $(PIPE_ARGS)

.PHONY: sync
sync: install ## Run all three pipelines: extract -> embed -> graph
	$(PY) -m rag.sync $(PIPE_ARGS)

.PHONY: graphrag
graphrag: install ## Index the Document Store with Microsoft GraphRAG (separate DB)
	$(PY) -m rag.graphrag $(PIPE_ARGS)

.PHONY: chat
chat: install ## Start the interactive chat REPL
	rlwrap $(PY) -m rag.chat --trace

VIEW_ARGS ?= --host 127.0.0.1 --port 5000 --reload

.PHONY: view
view: install ## Launch the interactive store viewer (Flask app)
	$(PY) -m rag.viewer $(VIEW_ARGS)

.PHONY: config
config: ## Create config.yaml from the example if missing
	@test -f config.yaml && echo "config.yaml already exists" \
		|| (cp config.yaml.example config.yaml && echo "created config.yaml — now edit it")

.PHONY: clean
clean: ## Remove Python caches and pytest cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

.PHONY: distclean
distclean: clean ## clean + remove the virtualenv
	rm -rf $(VENV)
