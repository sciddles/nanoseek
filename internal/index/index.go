// Package index implements nanoseek's search engine: a BM25 lexical ranker and
// a hashed-feature vector ranker, fused with reciprocal rank fusion. It has no
// dependencies outside the standard library and keeps everything in memory.
package index

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

// BM25 parameters. k1 controls how fast term frequency saturates and b how
// strongly long documents are penalized; these are the usual defaults.
const (
	bm25K1 = 1.5
	bm25B  = 0.75
)

// rrfK is the reciprocal-rank-fusion constant. A larger value flattens the
// curve, so the two rankers have to agree more before a document climbs.
const rrfK = 60.0

// Mode selects which ranker (or combination) a search uses.
type Mode string

const (
	ModeHybrid  Mode = "hybrid"
	ModeLexical Mode = "lexical"
	ModeVector  Mode = "vector"
)

// ParseMode maps a request string to a Mode, defaulting to hybrid.
func ParseMode(s string) (Mode, error) {
	switch Mode(strings.ToLower(strings.TrimSpace(s))) {
	case "", ModeHybrid:
		return ModeHybrid, nil
	case ModeLexical:
		return ModeLexical, nil
	case ModeVector:
		return ModeVector, nil
	default:
		return "", fmt.Errorf("unknown mode %q: want hybrid, lexical or vector", s)
	}
}

// Document is one indexed item. ID is assigned by the caller or generated on
// insert; Meta is free-form and is carried through to search results untouched.
type Document struct {
	ID    string            `json:"id"`
	Title string            `json:"title"`
	Text  string            `json:"text"`
	Meta  map[string]string `json:"meta,omitempty"`
}

// content is what actually gets indexed: the title participates in matching
// rather than being decoration.
func (d Document) content() string {
	if d.Title == "" {
		return d.Text
	}
	return d.Title + "\n" + d.Text
}

// Result is one hit. Lexical and Vector are the raw per-ranker scores, kept
// alongside the fused Score so the playground can show why a hit ranked here.
type Result struct {
	Document Document `json:"document"`
	Score    float64  `json:"score"`
	Lexical  float64  `json:"lexical"`
	Vector   float64  `json:"vector"`
	Snippet  string   `json:"snippet"`
}

// Stats is a snapshot of index size, used by the /api/stats endpoint.
type Stats struct {
	Documents int     `json:"documents"`
	Terms     int     `json:"terms"`
	AvgLength float64 `json:"avg_length"`
}

type posting struct {
	doc int
	tf  int
}

type entry struct {
	doc    Document
	vec    Vector
	length int
}

// Index is a safe-for-concurrent-use in-memory search index.
type Index struct {
	mu       sync.RWMutex
	entries  []*entry // nil at a slot means the document was deleted
	byID     map[string]int
	postings map[string][]posting
	live     int
	totalLen int
	nextID   int
}

// New returns an empty index.
func New() *Index {
	return &Index{
		byID:     make(map[string]int),
		postings: make(map[string][]posting),
	}
}

// Add indexes a document, replacing any existing one with the same ID. A blank
// ID gets an auto-generated one. It returns the ID actually used.
func (ix *Index) Add(doc Document) (string, error) {
	if strings.TrimSpace(doc.Text) == "" && strings.TrimSpace(doc.Title) == "" {
		return "", fmt.Errorf("document needs a title or text")
	}

	ix.mu.Lock()
	defer ix.mu.Unlock()

	if doc.ID == "" {
		doc.ID = ix.generateID()
	} else if slot, ok := ix.byID[doc.ID]; ok {
		ix.removeSlot(slot)
	}

	tokens := Tokenize(doc.content())
	e := &entry{doc: doc, vec: Embed(doc.content()), length: len(tokens)}

	slot := len(ix.entries)
	ix.entries = append(ix.entries, e)
	ix.byID[doc.ID] = slot
	ix.live++
	ix.totalLen += e.length

	tf := make(map[string]int, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}
	for term, n := range tf {
		ix.postings[term] = append(ix.postings[term], posting{doc: slot, tf: n})
	}
	return doc.ID, nil
}

// Delete removes a document by ID, reporting whether it was there.
func (ix *Index) Delete(id string) bool {
	ix.mu.Lock()
	defer ix.mu.Unlock()

	slot, ok := ix.byID[id]
	if !ok {
		return false
	}
	ix.removeSlot(slot)
	return true
}

// removeSlot drops a document and every posting pointing at it, so document
// frequencies stay exact instead of counting tombstones. Callers hold the lock.
func (ix *Index) removeSlot(slot int) {
	e := ix.entries[slot]
	if e == nil {
		return
	}
	for _, term := range uniqueTokens(e.doc.content()) {
		list := ix.postings[term]
		for i, p := range list {
			if p.doc == slot {
				ix.postings[term] = append(list[:i], list[i+1:]...)
				break
			}
		}
		if len(ix.postings[term]) == 0 {
			delete(ix.postings, term)
		}
	}
	delete(ix.byID, e.doc.ID)
	ix.entries[slot] = nil
	ix.live--
	ix.totalLen -= e.length
}

// Documents returns every live document, oldest first. The slice is a copy.
func (ix *Index) Documents() []Document {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	docs := make([]Document, 0, ix.live)
	for _, e := range ix.entries {
		if e != nil {
			docs = append(docs, e.doc)
		}
	}
	return docs
}

// Stats reports the current size of the index.
func (ix *Index) Stats() Stats {
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	s := Stats{Documents: ix.live, Terms: len(ix.postings)}
	if ix.live > 0 {
		s.AvgLength = float64(ix.totalLen) / float64(ix.live)
	}
	return s
}

// Search ranks documents against a query and returns the top k hits.
func (ix *Index) Search(query string, k int, mode Mode) []Result {
	if k <= 0 {
		k = 10
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	if ix.live == 0 || strings.TrimSpace(query) == "" {
		return []Result{}
	}

	lexical := ix.bm25(query)
	vector := ix.cosine(query)

	var fused map[int]float64
	switch mode {
	case ModeLexical:
		fused = lexical
	case ModeVector:
		fused = vector
	default:
		fused = fuse(rank(lexical), rank(vector))
	}

	slots := make([]int, 0, len(fused))
	for slot, score := range fused {
		if score > 0 {
			slots = append(slots, slot)
		}
	}
	// Ties break on document ID so that equal scores rank deterministically.
	sort.Slice(slots, func(a, b int) bool {
		if fused[slots[a]] != fused[slots[b]] {
			return fused[slots[a]] > fused[slots[b]]
		}
		return ix.entries[slots[a]].doc.ID < ix.entries[slots[b]].doc.ID
	})
	if len(slots) > k {
		slots = slots[:k]
	}

	results := make([]Result, 0, len(slots))
	for _, slot := range slots {
		e := ix.entries[slot]
		results = append(results, Result{
			Document: e.doc,
			Score:    fused[slot],
			Lexical:  lexical[slot],
			Vector:   vector[slot],
			Snippet:  snippet(e.doc.Text, query),
		})
	}
	return results
}

// bm25 scores every document that contains at least one query term. Callers
// hold the read lock.
func (ix *Index) bm25(query string) map[int]float64 {
	scores := make(map[int]float64)
	if ix.live == 0 {
		return scores
	}
	avgLen := float64(ix.totalLen) / float64(ix.live)
	if avgLen == 0 {
		return scores
	}
	n := float64(ix.live)

	for _, term := range Tokenize(query) {
		list := ix.postings[term]
		if len(list) == 0 {
			continue
		}
		df := float64(len(list))
		idf := math.Log(1 + (n-df+0.5)/(df+0.5))
		for _, p := range list {
			tf := float64(p.tf)
			dl := float64(ix.entries[p.doc].length)
			scores[p.doc] += idf * (tf * (bm25K1 + 1)) /
				(tf + bm25K1*(1-bm25B+bm25B*dl/avgLen))
		}
	}
	return scores
}

// cosine scores every live document against the query vector. This is a linear
// scan: fine for the thousands of documents this engine targets, and the place
// to bolt on an approximate index if the corpus outgrows it.
func (ix *Index) cosine(query string) map[int]float64 {
	qv := Embed(query)
	scores := make(map[int]float64, ix.live)
	for slot, e := range ix.entries {
		if e == nil {
			continue
		}
		if sim := Cosine(qv, e.vec); sim > 0 {
			scores[slot] = sim
		}
	}
	return scores
}

// rank converts scores into 1-based ranks, highest score first.
func rank(scores map[int]float64) map[int]int {
	slots := make([]int, 0, len(scores))
	for slot, s := range scores {
		if s > 0 {
			slots = append(slots, slot)
		}
	}
	sort.Slice(slots, func(a, b int) bool {
		if scores[slots[a]] != scores[slots[b]] {
			return scores[slots[a]] > scores[slots[b]]
		}
		return slots[a] < slots[b]
	})
	ranks := make(map[int]int, len(slots))
	for i, slot := range slots {
		ranks[slot] = i + 1
	}
	return ranks
}

// fuse combines rankings with reciprocal rank fusion. RRF only looks at
// positions, so the two rankers' incomparable score scales (unbounded BM25 vs
// cosine in [0,1]) never have to be normalized against each other.
func fuse(rankings ...map[int]int) map[int]float64 {
	fused := make(map[int]float64)
	for _, r := range rankings {
		for slot, pos := range r {
			fused[slot] += 1 / (rrfK + float64(pos))
		}
	}
	return fused
}

func uniqueTokens(text string) []string {
	seen := make(map[string]struct{})
	var out []string
	for _, t := range Tokenize(text) {
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	return out
}

func (ix *Index) generateID() string {
	for {
		ix.nextID++
		id := fmt.Sprintf("doc-%d", ix.nextID)
		if _, taken := ix.byID[id]; !taken {
			return id
		}
	}
}
