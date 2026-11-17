# Palm Mind – RAG Backend

FastAPI backend with two main APIs:

1. **Document Ingestion** – upload `.pdf` / `.txt`, extract text, chunk it with one of two strategies, embed, store vectors in **Qdrant** and metadata in a SQL database.
2. **Conversational RAG** – multi-turn chat over the ingested documents with **Redis** chat memory, plus LLM-driven **interview booking** (name, email, date, time) that is persisted to the database.

No LangChain / RetrievalQAChain, no FAISS/Chroma, no UI – the retrieval and prompting pipeline is written by hand.

## Stack

| Concern | Choice |
|---|---|
| API | FastAPI, Pydantic v2 |
| Text extraction | pypdf |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, normalised) |
| Vector DB | Qdrant (server, Qdrant Cloud, or embedded mode for dev) |
| Metadata / bookings | SQLAlchemy 2.0 – SQLite by default, any SQLAlchemy URL works |
| Chat memory | Redis |
| LLM | Groq (`llama-3.3-70b-versatile` by default) via the official `groq` SDK |

## Project layout

```
app/
  main.py               app factory, lifespan, service wiring
  core/                 settings, logging, exceptions
  db/                   engine/session + ORM models (Document, Booking)
  schemas/              Pydantic request/response models
  api/
    deps.py             DI helpers (services live on app.state)
    routes/documents.py ingestion endpoints
    routes/chat.py      chat, history, bookings endpoints
  services/
    text_extraction.py  pdf/txt -> clean text
    chunking.py         FixedSizeChunker, SemanticChunker
    embeddings.py       sentence-transformers wrapper
    vector_store.py     Qdrant wrapper (VectorStore protocol)
    ingestion.py        extract -> chunk -> embed -> upsert -> metadata
    chat_memory.py      Redis history + booking drafts
    llm.py              Groq SDK wrapper (text + validated JSON output)
    prompts.py          system prompts
    booking.py          booking validation + persistence
    chat.py             conversational RAG orchestration
tests/                  unit + API tests (no external services needed)
```

## Running it

### Option A – Docker Compose (everything local)

```bash
cp .env.example .env        # set GROQ_API_KEY
docker compose up --build
```

API docs: http://localhost:8000/docs

### Option B – local Python

Requires Python 3.12+, and Redis + Qdrant reachable somewhere (local, Docker, or the free tiers of Redis Cloud / Qdrant Cloud). If `QDRANT_URL` is empty the app uses embedded Qdrant on disk, so only Redis is strictly required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # fill in keys / URLs
uvicorn app.main:app --reload
```

The first request that needs embeddings downloads the MiniLM model (~90 MB).

## API

All endpoints are under `/api/v1`. Interactive docs at `/docs`.

### Ingestion

`POST /documents/upload` – multipart form

| field | type | notes |
|---|---|---|
| `file` | file | `.pdf` or `.txt`, max 20 MB (configurable) |
| `chunking_strategy` | `fixed` \| `semantic` | default `fixed` |

```bash
curl -F "file=@handbook.pdf" -F "chunking_strategy=semantic" \
     http://localhost:8000/api/v1/documents/upload
```

```json
{
  "document": {
    "id": "6f1c…",
    "filename": "handbook.pdf",
    "chunking_strategy": "semantic",
    "chunk_count": 42,
    "status": "ready",
    ...
  },
  "message": "Indexed 42 chunks using 'semantic' chunking."
}
```

`GET /documents` – list metadata, `GET /documents/{id}` – single document.

### Chat

`POST /chat`

```json
{ "session_id": null, "message": "What is the leave policy?" }
```

Omit `session_id` on the first message; reuse the one returned for follow-ups.

```json
{
  "session_id": "a3f9…",
  "intent": "question",
  "answer": "Employees get 18 days of paid leave per year …",
  "sources": [{ "filename": "handbook.pdf", "chunk_index": 7, "score": 0.71, "text": "…" }],
  "booking": null,
  "pending_booking_fields": []
}
```

Booking example (3 turns):

```
> I'd like to book an interview
< To book your interview I still need your full name, your email address, the preferred date and the preferred time.
> Sure, I'm Ram Sharma, ram@example.com
< Thanks Ram! To book your interview I still need the preferred date and the preferred time.
> next Monday at 3pm
< Your interview is booked, Ram Sharma! Monday, 23 November 2026 at 15:00. ...
```

Other endpoints: `GET /chat/{session_id}/history`, `DELETE /chat/{session_id}`, `GET /bookings?email=…`.

## How it works

### Chunking strategies

- **fixed** – sliding window over words (default 200 words, 40 overlap). Cheap and predictable; overlap keeps sentences that straddle a boundary retrievable from both sides.
- **semantic** – split text into sentences, embed each one, and start a new chunk whenever cosine similarity between neighbouring sentences drops below a threshold (default 0.55), i.e. when the topic shifts. A max word count stops chunks from growing unbounded. Gives more coherent chunks for documents with distinct sections, at the cost of embedding every sentence during ingestion.

Both implement the same `Chunker` protocol, so adding a strategy is one class plus an enum value.

### Ingestion pipeline

`extract text -> create Document row (processing) -> chunk -> embed -> upsert to Qdrant -> mark ready`

Point IDs are deterministic (`uuid5(document_id:chunk_index)`), and each payload carries `document_id`, `filename`, `chunk_index`, `text` and `strategy`. There's a payload index on `document_id` for filtering. If any step fails the row is marked `failed` with the error, so metadata never claims a document is searchable when it isn't.

### Conversational RAG

Each message goes through:

1. **Load state** – last 10 messages + any in-progress booking draft from Redis.
2. **Route** – one structured-output LLM call returns:
   - `intent`: `question` / `booking` / `chitchat`
   - `standalone_query`: the message rewritten with context resolved ("how much is it?" -> "price of the Pro plan"). This is what makes multi-turn retrieval work.
   - `booking`: any of name/email/date/time found in the message (relative dates resolved using today's date)
   - `cancel_booking`
3. **Question** – embed `standalone_query`, search Qdrant (top-k + score threshold), put the excerpts in the prompt with the conversation history, generate a grounded answer, return the sources.
4. **Booking** – merge extracted fields into the draft in Redis; ask for whatever is still missing. Once complete, validate with Pydantic (`EmailStr`, date not in the past, valid time). Invalid fields are dropped from the draft and asked for again; valid bookings are saved to the `bookings` table and the draft is cleared.
5. **Save** both turns to Redis (list trimmed to the last 20 messages, 24h TTL).

A question asked in the middle of a booking is answered normally and the user is reminded what's still missing.

The router uses Groq's JSON mode with the Pydantic schema in the system prompt. The output is validated with Pydantic, and if it doesn't match, the model gets one retry with the validation errors fed back. The router runs at temperature 0 so routing is deterministic.

### Redis keys

```
chat:{session_id}:messages   list of JSON messages (capped, TTL)
chat:{session_id}:booking    hash with the partial booking (TTL)
```

## Tests and checks

```bash
pytest            # uses fakeredis, an in-memory vector store and a scripted LLM
ruff check app tests
mypy app          # strict mode
```

## Things I'd do next

- Move ingestion to a background worker (Celery/RQ/Arq) for large PDFs and return a job id.
- Hybrid search (BM25 + vectors) and a re-ranker for better retrieval on keyword-heavy questions.
- Check booking slot availability / double bookings and send a confirmation email.
- Alembic migrations instead of `create_all`.
- Auth and per-user document isolation (filter by owner in Qdrant).
- OCR for scanned PDFs.
