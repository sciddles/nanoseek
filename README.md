# nanoseek

A tiny hybrid search engine in one Go binary: BM25 for the words people actually
typed, hashed-feature vectors for the words they meant, and reciprocal rank
fusion to settle the argument. No dependencies outside the standard library, no
API keys, no database, no network calls.

```
$ nanoseek query "how should I combine two rankers"
#  SCORE   BM25    COSINE  TITLE
1  0.0328  10.835  0.245   Reciprocal rank fusion, the cheapest way to combine rankers
2  0.0323  2.945   0.147   Cosine similarity ignores how loud you are
3  0.0315  2.631   0.131   BM25, the ranking function that refuses to retire
```

## Why both rankers

Lexical search is excellent at exact tokens — error codes, identifiers, rare
proper nouns — and helpless when the question is worded differently from the
document. Vector search handles the paraphrase and fumbles the serial number you
pasted in. Each one's failure mode is the other's strength, so nanoseek runs both
and fuses the two ranked lists:

```
score(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_cosine(d))
```

Reciprocal rank fusion throws the raw scores away and keeps only positions, which
is what makes it safe to combine an unbounded BM25 score with a cosine similarity
bounded in [0, 1] — no normalisation constant to tune, no weight to guess.

Every response carries all three numbers, so you can always see which ranker put
a result where it is.

## Quickstart

Needs Go 1.24 or newer. Nothing else.

```bash
git clone https://github.com/sciddles/nanoseek.git
cd nanoseek
go run ./cmd/nanoseek serve
```

Open <http://localhost:8080> for the playground: search box, a toggle between
hybrid, lexical and vector, per-hit score breakdowns, and a form to add documents
to the live index. It ships with a twenty-document corpus about retrieval so
there is something to search on the first run.

From the terminal instead:

```bash
go run ./cmd/nanoseek query "chunking strategy" -k 3
go run ./cmd/nanoseek query "tokenisation" -mode vector
go run ./cmd/nanoseek add -title "Meeting notes" -text "We agreed to ship on Friday."
go run ./cmd/nanoseek stats
```

Or build the single binary — HTML included — and hand it to someone:

```bash
go build -o nanoseek ./cmd/nanoseek
./nanoseek serve -addr :3000 -data data/corpus.jsonl
```

## HTTP API

| Method   | Path                    | Body / query                                  |
| -------- | ----------------------- | --------------------------------------------- |
| `GET`    | `/api/search`           | `?q=…&mode=hybrid` \| `lexical` \| `vector`, `&k=5`         |
| `POST`   | `/api/search`           | `{"query": "…", "mode": "hybrid", "k": 5}`    |
| `GET`    | `/api/documents`        | —                                             |
| `POST`   | `/api/documents`        | `{"id": "…", "title": "…", "text": "…"}`      |
| `DELETE` | `/api/documents/{id}`   | —                                             |
| `GET`    | `/api/stats`            | —                                             |

```bash
curl -s 'localhost:8080/api/search?q=prompt+injection&k=2' | jq '.results[].document.title'

curl -s localhost:8080/api/documents \
  -H 'Content-Type: application/json' \
  -d '{"title":"Standup", "text":"Blocked on the index rebuild."}'
```

`id` is optional on insert and generated when absent; posting an existing `id`
replaces that document. Writes are persisted to the JSONL corpus after every
change, written to a temp file and renamed so an interrupted write cannot corrupt
the corpus.

## How it works

```
                    ┌──────────────┐
      query ───────▶│  BM25 ranker │──── ranked list ──┐
        │           └──────────────┘                   │
        │                                              ▼
        │                                     ┌─────────────────┐
        │                                     │ reciprocal rank │──▶ results
        │                                     │     fusion      │
        │           ┌──────────────┐          └─────────────────┘
        └──────────▶│ hashed-vector│──── ranked list ──┘
                    │    cosine    │
                    └──────────────┘
```

**Lexical half** — an inverted index with BM25 (`k1=1.5`, `b=0.75`): term
frequency saturates so repetition stops paying, and long documents are penalised
against the collection average.

**Vector half** — `Embed` hashes word tokens and padded character 3-grams into
32k buckets, applies sublinear scaling, and L2-normalises. The n-grams are what
let a search for `normalise` find a document that only says `normalisation`. This
is a *lexical* similarity dressed as a vector, not a learned embedding — see
below for swapping in a real one.

**Storage** — everything lives in memory behind a `sync.RWMutex`; the corpus is a
JSON Lines file. Cosine scoring is a linear scan over the corpus, which is
instant at the thousands-of-documents scale this targets.

| File | What lives there |
| ---- | ---------------- |
| [`internal/index/index.go`](internal/index/index.go) | inverted index, BM25, RRF, search |
| [`internal/index/vector.go`](internal/index/vector.go) | the hashing-trick embedding |
| [`internal/index/tokenize.go`](internal/index/tokenize.go) | tokenizer and character n-grams |
| [`internal/index/store.go`](internal/index/store.go) | JSONL load and atomic save |
| [`internal/server/server.go`](internal/server/server.go) | HTTP API |
| [`internal/server/static/index.html`](internal/server/static/index.html) | the playground, one file, no build step |
| [`cmd/nanoseek/main.go`](cmd/nanoseek/main.go) | CLI |

## Where to take it next

The seams are deliberate. Each of these is a contained change:

- **Real embeddings.** Replace the body of `Embed` with a call to an embedding
  API and keep everything else. The fusion step does not care where the vectors
  came from — and it is the reason a swapped-in model can be evaluated against
  the current one honestly.
- **Answer generation.** `/api/search` already returns ranked passages, which is
  the retrieval half of a RAG stack. Add an endpoint that stuffs the top hits
  into a prompt and returns a grounded answer with citations.
- **A reranker.** Retrieve fifty, rerank the top five with a cross-encoder.
- **Approximate nearest neighbours.** `Index.cosine` is the linear scan to
  replace with HNSW or IVF once the corpus outgrows memory bandwidth.
- **Chunking.** Split long documents into overlapping windows at insert time and
  carry a parent id in `Meta`.
- **Evaluation.** Twenty questions with an expected document id each, and a
  `recall@k` command. Retrieval quality caps everything built above it, so this
  is worth more than it looks.

If you index documents you did not write — scraped pages, user uploads — treat
retrieved text as data, never as instructions to a downstream model. The corpus
that ships here has a document about exactly that.

## Tests

```bash
go test ./...          # unit and HTTP tests
go test -race ./...    # the index is exercised concurrently
```

The suite covers tokenisation, vector normalisation, ranking order in all three
modes, deletion and posting cleanup, snippet extraction, JSONL round-tripping,
API validation, and concurrent read/write traffic against the index.

## License

MIT — see [LICENSE](LICENSE).
