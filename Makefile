# Convenience targets for the RAG pipeline. Each stage caps at LIMIT documents.
# Usage: make extract | make translate | make embed | make browse
#        make embed MODEL=bm25    # embed a different model

RAG        := ./rag-cli
LIMIT      ?= 10
OLLAMA_URL ?= http://localhost:11434

.PHONY: extract extract_full delete translate embed browse pipeline ollama

## Extract up to LIMIT emails: all of the 'private' account, plus the INBOX of
## 'rwth', 'ut' and 'qis-office'. Additive only — no deletions here (a message
## moved out of a synced folder is never mistaken for a deletion). Run `make
## delete` to reconcile removals.
extract:
	$(RAG) --verbose extract email account=private count=$(LIMIT)
	$(RAG) --verbose extract email account=rwth folder=INBOX count=$(LIMIT)
	$(RAG) --verbose extract email account=ut folder=INBOX count=$(LIMIT)
	$(RAG) --verbose extract email account=qis-office folder=INBOX count=$(LIMIT)

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
embed: ollama
	$(RAG) --verbose embed --model bm25 # --limit $(LIMIT)
	$(RAG) --verbose embed --model qwen3-8b # --limit $(LIMIT)

## Launch the db-browser web UI.
browse:
	$(RAG) maintenance db-browser

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
	  setsid ollama serve >/tmp/ollama-serve.log 2>&1 </dev/null & \
	  for i in $$(seq 1 30); do \
	    curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 && break; \
	    sleep 1; \
	  done; \
	  curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 || { \
	    echo "error: Ollama did not come up (see /tmp/ollama-serve.log)"; exit 1; }; \
	}
