# Convenience targets for the RAG pipeline. Each stage caps at LIMIT documents.
# Usage: make extract | make translate | make embed | make browse
#        make embed MODEL=bm25    # embed a different model

RAG        := ./rag-cli
LIMIT      ?= 100
MODEL      ?= qwen3-8b
OLLAMA_URL ?= http://localhost:11434

.PHONY: extract translate embed browse ollama

## Extract up to LIMIT emails: all of the 'private' account, plus 'rwth' INBOX.
extract:
	$(RAG) --verbose extract email account=private count=$(LIMIT)
	$(RAG) --verbose extract email account=rwth folder=INBOX count=$(LIMIT)

## Translate up to LIMIT documents to Markdown.
translate:
	$(RAG) --verbose translate --limit $(LIMIT)

## Embed up to LIMIT documents into MODEL's index (needs the Ollama server).
embed: ollama
	$(RAG) --verbose embed --model $(MODEL) --limit $(LIMIT)

## Launch the db-browser web UI.
browse:
	$(RAG) maintenance db-browser

## Ensure an Ollama server is reachable; start one in the background if not.
ollama:
	@curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 || { \
	  echo "Ollama not reachable at $(OLLAMA_URL); starting 'ollama serve'..."; \
	  ollama serve >/tmp/ollama-serve.log 2>&1 & \
	  for i in $$(seq 1 30); do \
	    curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 && break; \
	    sleep 1; \
	  done; \
	  curl -sf $(OLLAMA_URL)/api/tags >/dev/null 2>&1 || { \
	    echo "error: Ollama did not come up (see /tmp/ollama-serve.log)"; exit 1; }; \
	}
