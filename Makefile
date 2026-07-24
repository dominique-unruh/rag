# Convenience targets for the RAG pipeline. Each stage caps at LIMIT documents.
# Usage: make extract | make translate | make embed | make browse
#        make embed MODEL=bm25    # embed a different model

RAG        := ./rag-cli
LIMIT      ?= 10
SINCE      := 2026-07-01
OLLAMA_URL ?= http://localhost:11434
# Cap concurrent request contexts to 1: single-user serial embedding never needs
# more, and each extra context multiplies peak RAM (guards against swap).
OLLAMA_NUM_PARALLEL ?= 1
# Config file the helper targets read (storage dir, vector-store URL) via yq.
CONFIG     ?= config.yaml
# Container image + name for the local Qdrant server target.
QDRANT_IMAGE ?= qdrant/qdrant
QDRANT_NAME  ?= rag-qdrant

.PHONY: extract extract_full delete translate embed browse chat pipeline ollama qdrant-server qdrant-server-stop

## Extract up to LIMIT emails: all of the 'private' account, plus the INBOX of
## 'rwth', 'ut' and 'qis-office'. Additive only — no deletions here (a message
## moved out of a synced folder is never mistaken for a deletion). Run `make
## delete` to reconcile removals.
extract:
	$(RAG) --verbose extract email account=private "since=$(SINCE)"
	$(RAG) --verbose extract email account=rwth "since=$(SINCE)"
	$(RAG) --verbose extract email account=ut "since=$(SINCE)"
	$(RAG) --verbose extract email account=qis-office "since=$(SINCE)"

## Like `extract`, but a full resync (full=true): re-fetch every message,
## ignoring the stored per-folder UID memory.
extract_full: delete
	$(RAG) --verbose extract email account=private count=$(LIMIT) full=true
	$(RAG) --verbose extract email account=rwth folder=INBOX count=$(LIMIT) full=true
	$(RAG) --verbose extract email account=ut folder=INBOX count=$(LIMIT) full=true
	$(RAG) --verbose extract email account=qis-office folder=INBOX count=$(LIMIT) full=true

## Reconcile deletions (delete=true): scan EVERY account's folders and remove
## documents whose message-id survives nowhere. Expensive (connects to all
## accounts, may fetch headers) and global, so it ignores account/folder/count.
delete:
	$(RAG) --verbose extract email delete=true

## Translate up to LIMIT documents to Markdown.
translate:
	$(RAG) --verbose translate --limit $(LIMIT)

## Embed up to LIMIT documents into MODEL's index (needs the Ollama server).
embed: ollama qdrant-server
	$(RAG) --verbose embed --model bm25 # --limit $(LIMIT)
	$(RAG) --verbose embed --model qwen3-8b # --limit $(LIMIT)

## Launch the db-browser web UI.
browse: ollama qdrant-server
	$(RAG) maintenance db-browser

## Interactive chat over the RAG (simple mode). Needs the Ollama server (chat
## model) and, in server mode, the Qdrant server for retrieval.
chat: ollama qdrant-server
	$(RAG) chat

## Run the full pipeline. Every stage runs even if an earlier one fails, but
## the target exits non-zero if any stage failed.
pipeline:
	@rc=0; \
	$(MAKE) extract   || rc=1; \
	$(MAKE) translate || rc=1; \
	$(MAKE) embed     || rc=1; \
	exit $$rc


## Ensure an Ollama server is reachable; start one in the background if not.
ollama:
	@curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 || { \
	  echo "Ollama not reachable at $(OLLAMA_URL); starting 'ollama serve'..."; \
	  setsid env OLLAMA_NUM_PARALLEL=$(OLLAMA_NUM_PARALLEL) ollama serve >/tmp/ollama-serve.log 2>&1 </dev/null & \
	  for i in $$(seq 1 30); do \
	    curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 && break; \
	    sleep 1; \
	  done; \
	  curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 || { \
	    echo "error: Ollama did not come up (see /tmp/ollama-serve.log)"; exit 1; }; \
	}

## Start a local Qdrant server (Docker) with an API key, data under the configured
## storage dir. Reads storage.dir + vector_store.url from $(CONFIG) via the Python
## yq (kislyuk/yq, a jq wrapper — needs jq; guarded below), the API key from the
## keyring (secret 'vector_store_api_key', service 'rag'). Idempotent: a no-op if
## the container is already running. Stop it with `make qdrant-server-stop`.
qdrant-server:
	@command -v yq >/dev/null 2>&1     || { echo "error: yq not installed"; exit 1; }
	@yq --help 2>&1 | grep -qiE 'jq wrapper|kislyuk' \
	  || { echo "error: this target needs the Python yq (kislyuk/yq, a jq wrapper); a different 'yq' (e.g. mikefarah) is on PATH"; exit 1; }
	@command -v jq >/dev/null 2>&1     || { echo "error: jq not installed (required by python yq)"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "error: docker not installed"; exit 1; }
	@if [ -n "$$(docker ps -q -f name=^/$(QDRANT_NAME)$$)" ]; then \
	   echo "Qdrant already running (container '$(QDRANT_NAME)')."; exit 0; fi; \
	 STORAGE="$$(yq -r '.storage.dir // "./storage"' $(CONFIG))"; \
	 URL="$$(yq -r '.vector_store.url // "http://localhost:6333"' $(CONFIG))"; \
	 HOST="$$(printf '%s' "$$URL" | sed -E 's#^[a-zA-Z]+://([^:/]+).*#\1#')"; \
	 [ -n "$$HOST" ] || HOST=localhost; \
	 PORT="$$(printf '%s' "$$URL" | sed -E 's#^.*://[^:/]+:?([0-9]*).*#\1#')"; \
	 case "$$PORT" in ''|*[!0-9]*) PORT=6333;; esac; \
	 GPORT=$$((PORT + 1)); \
	 QDIR="$$(realpath -m "$$STORAGE")/qdrant-server"; \
	 SECRET="vector_store_api_key:$$HOST:$$PORT"; \
	 PYBIN=".venv/bin/python"; [ -x "$$PYBIN" ] || PYBIN="python3"; \
	 KEY="$$(PYTHON_KEYRING_BACKEND=keyring.backends.SecretService.Keyring \
	         $$PYBIN -m keyring get rag "$$SECRET" 2>/dev/null || true)"; \
	 [ -n "$$KEY" ] || { echo "error: no API key in keyring; set one with: keyring set rag '$$SECRET'"; exit 1; }; \
	 mkdir -p "$$QDIR"; \
	 docker rm -f $(QDRANT_NAME) >/dev/null 2>&1 || true; \
	 echo "Starting Qdrant on port $$PORT (grpc $$GPORT), storage at $$QDIR ..."; \
	 env QDRANT__SERVICE__API_KEY="$$KEY" docker run -d --name $(QDRANT_NAME) \
	   -p $$PORT:6333 -p $$GPORT:6334 \
	   -e QDRANT__SERVICE__API_KEY \
	   -v "$$QDIR:/qdrant/storage" \
	   $(QDRANT_IMAGE) >/dev/null; \
	 echo "Qdrant started (container '$(QDRANT_NAME)'). Stop: make qdrant-server-stop"

## Stop and remove the local Qdrant server container.
qdrant-server-stop:
	@docker rm -f $(QDRANT_NAME) >/dev/null 2>&1 && echo "Stopped '$(QDRANT_NAME)'." \
	  || echo "No '$(QDRANT_NAME)' container running."
