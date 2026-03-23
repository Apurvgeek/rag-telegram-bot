# Mini-RAG Telegram Bot

A lightweight Retrieval-Augmented Generation (RAG) bot for Telegram.  
Ask questions in plain English — the bot retrieves relevant passages from your document library and generates grounded answers using a local or cloud LLM.

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd rag_telegram_bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set TELEGRAM_BOT_TOKEN
```

Get a bot token from [@BotFather](https://t.me/BotFather) on Telegram (`/newbot`).

### 3. Add your documents

Drop `.txt` or `.md` files into the `knowledge_base/` folder.  
Three sample files are included (return policy, tech FAQ, onboarding guide).

### 4. Ingest documents

```bash
python3 -m src.ingest
```

This embeds all documents and stores them in `rag_store.db`.  
Re-run whenever you add or update documents — the ingester skips unchanged chunks.

### 5. Start the LLM (Ollama, recommended)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama serve        # runs on localhost:11434
```

### 6. Run the bot

```bash
python3 bot.py
```

Open Telegram, find your bot, and start asking questions.

---

## Docker Compose (all-in-one)

Starts the bot and an Ollama instance together:

```bash
cp .env.example .env   # set TELEGRAM_BOT_TOKEN
docker compose up --build

# In a separate terminal, pull the model once:
docker compose exec ollama ollama pull llama3
```

---

## LLM Backends

Set `LLM_BACKEND` in `.env`:

| Value          | Requirement                    | Notes                              |
|----------------|--------------------------------|------------------------------------|
| `ollama`       | Ollama running locally         | Default. Zero API cost.            |
| `openai`       | `OPENAI_API_KEY` set in `.env` | Use `gpt-4o-mini` for speed/cost.  |
| `huggingface`  | `transformers` + `torch`       | Uncomment in `requirements.txt`.   |

---

## Commands

| Command               | Description                        |
|-----------------------|------------------------------------|
| `/start`              | Welcome message                    |
| `/help`               | Show usage instructions            |
| `/ask <question>`     | Query the knowledge base           |
| Plain text            | Treated as an `/ask` query         |

---

## Configuration Reference

All settings live in `.env`:

| Variable        | Default              | Description                                |
|-----------------|----------------------|--------------------------------------------|
| `KNOWLEDGE_DIR` | `knowledge_base`     | Folder scanned for `.txt` / `.md` files    |
| `DB_PATH`       | `rag_store.db`       | SQLite database path                       |
| `EMBED_MODEL`   | `all-MiniLM-L6-v2`   | sentence-transformers model name           |
| `CHUNK_SIZE`    | `400`                | Characters per chunk                       |
| `CHUNK_OVERLAP` | `80`                 | Overlap between consecutive chunks        |
| `TOP_K`         | `3`                  | Chunks retrieved per query                 |
| `MIN_SCORE`     | `0.20`               | Minimum cosine similarity threshold        |
| `HISTORY_LENGTH`| `3`                  | Conversation turns retained per user       |
| `SHOW_SOURCES`  | `true`               | Append source filenames to answers         |

---

## System Design

```
User (Telegram)
      │  /ask <query>
      ▼
┌─────────────────┐
│  bot.py          │  python-telegram-bot Application
│  handlers.py     │  CommandHandler / MessageHandler
└────────┬────────┘
         │ query
         ▼
┌─────────────────┐       ┌──────────────────────┐
│  cache.py        │──────▶│  retriever.py         │
│  LRU(256)        │       │  SentenceTransformer  │
└─────────────────┘       │  cosine similarity    │
                           │  numpy @ operator     │
                           └──────────┬───────────┘
                                      │ SQL SELECT
                                      ▼
                           ┌──────────────────────┐
                           │  rag_store.db         │
                           │  SQLite               │
                           │  (chunks + embeddings)│
                           └──────────────────────┘
         │ context chunks
         ▼
┌─────────────────┐
│  history.py      │  last-N turns per user_id
└────────┬────────┘
         │ prompt = context + history + query
         ▼
┌─────────────────┐
│  llm_client.py   │  Ollama / OpenAI / HuggingFace
└────────┬────────┘
         │ answer
         ▼
    Telegram reply
    (answer + sources)
```

### Ingestion pipeline (run once)

```
knowledge_base/*.md  ──▶  ingest.py
                              │
                         _split()        sliding-window chunker
                              │
                    SentenceTransformer  all-MiniLM-L6-v2
                              │
                         SQLite INSERT   rag_store.db
```

---

## Enhancements Implemented

- **Conversation history** — last 3 turns per user kept in memory, sent to LLM for context continuity
- **Query-level LRU cache** — identical queries skip re-embedding (256-entry cache)
- **Source snippets** — each answer shows which document(s) it drew from
- **Multi-backend LLM** — swap between Ollama, OpenAI, or HuggingFace with one env var
- **Incremental ingestion** — SHA-256 hash check means re-running ingest only processes changed chunks

---

## Project Structure

```
rag_telegram_bot/
├── bot.py                  # Entry point — registers handlers, starts polling
├── src/
│   ├── config.py           # Environment loading, all constants
│   ├── handlers.py         # /start /help /ask  Telegram command handlers
│   ├── ingest.py           # Document chunking + embedding pipeline
│   ├── retriever.py        # Cosine similarity search over SQLite embeddings
│   ├── llm_client.py       # Ollama / OpenAI / HuggingFace LLM wrapper
│   ├── history.py          # Per-user conversation history manager
│   └── cache.py            # LRU query cache
├── knowledge_base/
│   ├── return_policy.md
│   ├── tech_faq.md
│   └── onboarding_guide.md
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Evaluation Notes

### Model choice rationale

- **Embedding**: `all-MiniLM-L6-v2` — 22 MB, fast on CPU, strong semantic similarity for English text. No GPU required.
- **LLM (default)**: Llama 3 via Ollama — runs locally, no API cost, strong instruction-following. Swap to `gpt-4o-mini` for better quality on complex questions.

### Design decisions

- **SQLite over a vector DB** — keeps dependencies minimal and portable. Adequate for up to ~50k chunks. Migrate to `sqlite-vec` or ChromaDB for larger corpora.
- **Numpy cosine search** — loads all embeddings into RAM once (typical footprint < 50 MB for a 1000-chunk corpus). For larger datasets, batch or use an HNSW index.
- **Chunking strategy** — paragraph-aware splitter preserves semantic units better than fixed-character splits, reducing mid-sentence breaks that confuse the retriever.
