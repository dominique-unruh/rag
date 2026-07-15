Pipeline for processing emails, chats, misc files, document repos into a RAG.

Pipeline consists of several modules.

Module 1: 

Extraction. Extracts data from various sources.
Quite similar to the extraction in the prototype implementation.
Should have a DB that stores:
- A source type ID (e.g., `email`), refers to the extraction plugin
- A document ID (unique within one source type)
- Version (counter, starts with 1, must be increases if the source changes)
- Metadata (key/value)
  - Some general keys (e.g., date, title, mime-type)
  - Arbitrary plugin specific ones.
- The document source (blob).

Extraction should have a general part (handles DB, command line, etc.),
and a plugin system.
Each plugin should just be an implementation of a class `Extractor` (defined by the general part),
and it should be located in a separate subdir of the extraction package.
Plugin does not connect to the DB directly, the Extractor class is the only interface to the overall system.

Also: I would like blobs to be stored separately on the FS. (Not inside the DB). To be more friendly for incremental backups etc.

Extraction plugins should also be able to delete items, or update items (when changed).

Extraction plugins never touch documents related to other plugins.
(Enforced by the API.)

Module 2:

Processing of the extracted data.
Takes a document from extracted document store, and translates it into a markdown document.
This can involve:
- Dataformat conversion (e.g., email to text)
- Image processing (OCR/picture recognition)
- AI rewriting (e.g., notes can read by AI, and rewritten in a more self-contained form for better embedding)
- Can access the already produced RAG via AI. E.g., when rewriting terse notes, can use general information
  to understand e.g. what a given first name could refer to, or what the overall context is.
- Should also have access to the raw extraction results etc.
  Processing can depend on the document type, the document source, the age, etc. (Arbitrary filters).
- Identity translation can also be possible (e.g., if the source is already a plain text or markdown document)

Result should be stored into the same DB (different table, of course).

Any translated document should produce a row with:
- ID of source (source type, document ID)
- Metadata (produced by the processing); both some data resulting from extraction process, as well as 
  some information about what pipeline we used (e.g., which LLM we used, whether we did a 
  quick translate or using context, version of prompt, ...).
  Can also be useful for selecting which documents we retranslate optionally.
- Version of the source (taken from the source database, to find out whether something changed and needs reprocessing)
- Translation (blob), in FS.
- Translation date/time.
- Git revision of this program at time of translation.
- "is active"

Explanation of "is active": we allow to keep old translations in the DB (e.g., if we ever want to revert).
"is active" tells us whether this is the currently used translation.
Constraint: for every source ID that occurs in the translation DB: exactly one active. 

Translations can only be added to DB if the repo is fully committed (so that git revision entry is meaningful).
(But running translation without adding can be possible for inspection.)

Module 3:

Put processed data into a RAG. Chunking + embedding.
Will need to automatically update things if the translation changed.

Module 4:

Chat interface for RAG access.
Different modes and LLM models.
- Simple mode where the LLM is just given search results based on my query.
- Complex mode where LLM can access the RAG and documents and translations and so on and do web searches etc.
  LLM can decide how to plan the search.

General notes:

Blobs stored in FS via hash value to avoid duplication, referenced from DB via it.

Command line tools:
- extract
  - run individual source plugins
  - add plugin specific options
- translate
  - run translation on a subset of documents
    - specify various filters for selecting
    - specify max number of documents
    - by age of the last translation
    - more as we go along
  - inspection mode: output the result of translation into text file
  - dry run mode: don't store in DB (allowed even if repo is dirty).
- embed
  - filter
- chat
  - specify model and mode on command line
- maintainance:
  - run GC-like tasks (based on filters for example)
  - stats
  - new things as we go along

Configuration: central YAML; contains everything except secrets.
Secrets via keyring. (Like in prototype)