package index

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

func build(t *testing.T, docs ...Document) *Index {
	t.Helper()
	ix := New()
	for _, d := range docs {
		if _, err := ix.Add(d); err != nil {
			t.Fatalf("Add(%q): %v", d.ID, err)
		}
	}
	return ix
}

var corpus = []Document{
	{ID: "a", Title: "BM25 ranking", Text: "BM25 scores documents by term frequency with length normalisation."},
	{ID: "b", Title: "Cosine similarity", Text: "Cosine similarity compares the angle between two vectors."},
	{ID: "c", Title: "Baking bread", Text: "A sourdough starter needs flour, water and patience."},
}

func TestTokenizeSplitsOnNonAlphanumerics(t *testing.T) {
	got := Tokenize("GPU-bound (fast!) 42 — Алматы")
	want := []string{"gpu", "bound", "fast", "42", "алматы"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("Tokenize() = %v, want %v", got, want)
	}
}

func TestEmbedIsUnitLength(t *testing.T) {
	v := Embed("hybrid search fuses two rankers")
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	if math.Abs(norm-1) > 1e-9 {
		t.Errorf("squared norm = %v, want 1", norm)
	}
}

func TestCosineRanksRelatedTextHigher(t *testing.T) {
	query := Embed("vector similarity")
	related := Cosine(query, Embed("Cosine similarity compares vectors."))
	unrelated := Cosine(query, Embed("A sourdough starter needs flour."))
	if related <= unrelated {
		t.Errorf("related %v should outscore unrelated %v", related, unrelated)
	}
	if self := Cosine(query, query); math.Abs(self-1) > 1e-9 {
		t.Errorf("self similarity = %v, want 1", self)
	}
}

func TestSearchFindsTheObviousDocument(t *testing.T) {
	ix := build(t, corpus...)
	for _, mode := range []Mode{ModeHybrid, ModeLexical, ModeVector} {
		results := ix.Search("bm25 term frequency", 3, mode)
		if len(results) == 0 {
			t.Fatalf("%s: no results", mode)
		}
		if got := results[0].Document.ID; got != "a" {
			t.Errorf("%s: top hit = %q, want %q", mode, got, "a")
		}
	}
}

// The vector ranker matches on character n-grams, so a query that shares no
// whole word with the document should still retrieve it.
func TestVectorModeMatchesInflectedForms(t *testing.T) {
	ix := build(t, Document{ID: "x", Title: "Normalising vectors", Text: "Normalisation keeps comparisons fair."})
	results := ix.Search("normalise", 3, ModeVector)
	if len(results) == 0 {
		t.Fatal("expected a fuzzy match, got none")
	}
	if results[0].Document.ID != "x" {
		t.Errorf("top hit = %q, want %q", results[0].Document.ID, "x")
	}
}

func TestLexicalModeIgnoresDocumentsWithoutTheTerm(t *testing.T) {
	ix := build(t, corpus...)
	for _, r := range ix.Search("sourdough", 10, ModeLexical) {
		if r.Document.ID != "c" {
			t.Errorf("unexpected lexical hit %q", r.Document.ID)
		}
	}
}

func TestSearchRespectsK(t *testing.T) {
	ix := build(t, corpus...)
	if results := ix.Search("similarity documents starter", 2, ModeHybrid); len(results) > 2 {
		t.Errorf("got %d results, want at most 2", len(results))
	}
}

func TestEmptyQueryAndEmptyIndexReturnNoResults(t *testing.T) {
	if got := New().Search("anything", 5, ModeHybrid); len(got) != 0 {
		t.Errorf("empty index returned %d results", len(got))
	}
	if got := build(t, corpus...).Search("   ", 5, ModeHybrid); len(got) != 0 {
		t.Errorf("blank query returned %d results", len(got))
	}
}

func TestAddGeneratesIDsAndReplacesByID(t *testing.T) {
	ix := New()
	id, err := ix.Add(Document{Title: "No id here"})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if id == "" {
		t.Fatal("Add returned an empty id")
	}

	if _, err := ix.Add(Document{ID: id, Title: "Replaced", Text: "Entirely new text about kayaks."}); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if n := ix.Stats().Documents; n != 1 {
		t.Errorf("documents = %d, want 1 after replacing", n)
	}
	if got := ix.Search("kayaks", 1, ModeLexical); len(got) != 1 {
		t.Errorf("replacement text is not searchable: %d hits", len(got))
	}
	if got := ix.Search("no id here", 5, ModeLexical); len(got) != 0 {
		t.Errorf("old text is still indexed: %d hits", len(got))
	}
}

func TestAddRejectsEmptyDocuments(t *testing.T) {
	if _, err := New().Add(Document{Meta: map[string]string{"topic": "nothing"}}); err == nil {
		t.Error("expected an error for a document with no title or text")
	}
}

func TestDeleteRemovesDocumentAndItsPostings(t *testing.T) {
	ix := build(t, corpus...)
	if !ix.Delete("c") {
		t.Fatal("Delete reported the document was missing")
	}
	if ix.Delete("c") {
		t.Error("second Delete should report false")
	}
	if n := ix.Stats().Documents; n != 2 {
		t.Errorf("documents = %d, want 2", n)
	}
	if got := ix.Search("sourdough", 5, ModeHybrid); len(got) != 0 {
		t.Errorf("deleted document still matches: %d hits", len(got))
	}
	for _, d := range ix.Documents() {
		if d.ID == "c" {
			t.Error("Documents still lists the deleted document")
		}
	}
}

func TestSnippetCentresOnTheMatch(t *testing.T) {
	filler := strings.Repeat("padding words that carry no signal at all. ", 20)
	ix := build(t, Document{ID: "s", Title: "Long one", Text: filler + "the needle is buried here. " + filler})
	results := ix.Search("needle", 1, ModeLexical)
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if !strings.Contains(results[0].Snippet, "needle") {
		t.Errorf("snippet missed the match: %q", results[0].Snippet)
	}
}

func TestJSONLRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "corpus.jsonl")
	if err := build(t, corpus...).SaveJSONL(path); err != nil {
		t.Fatalf("SaveJSONL: %v", err)
	}

	reloaded := New()
	n, err := reloaded.LoadJSONL(path)
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	if n != len(corpus) {
		t.Errorf("loaded %d documents, want %d", n, len(corpus))
	}
	if got := reloaded.Search("cosine", 1, ModeLexical); len(got) != 1 || got[0].Document.ID != "b" {
		t.Errorf("reloaded index does not search correctly: %+v", got)
	}
}

func TestLoadMissingFileIsNotAnError(t *testing.T) {
	n, err := New().LoadJSONL(filepath.Join(t.TempDir(), "absent.jsonl"))
	if err != nil || n != 0 {
		t.Errorf("LoadJSONL(missing) = %d, %v; want 0, nil", n, err)
	}
}

func TestReadJSONLReportsTheBadLine(t *testing.T) {
	_, err := New().ReadJSONL(strings.NewReader("{\"id\":\"ok\",\"text\":\"fine\"}\n\nnot json\n"))
	if err == nil || !strings.Contains(err.Error(), "line 3") {
		t.Errorf("error = %v, want it to name line 3", err)
	}
}

func TestParseMode(t *testing.T) {
	for in, want := range map[string]Mode{"": ModeHybrid, "Hybrid": ModeHybrid, "lexical": ModeLexical, " vector ": ModeVector} {
		got, err := ParseMode(in)
		if err != nil || got != want {
			t.Errorf("ParseMode(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := ParseMode("magic"); err == nil {
		t.Error("expected an error for an unknown mode")
	}
}

func TestConcurrentSearchAndWrite(t *testing.T) {
	ix := build(t, corpus...)
	done := make(chan struct{})
	go func() {
		for i := 0; i < 200; i++ {
			ix.Search("cosine vectors", 3, ModeHybrid)
		}
		close(done)
	}()
	for i := 0; i < 200; i++ {
		if _, err := ix.Add(Document{Title: "churn", Text: "documents come and go"}); err != nil {
			t.Errorf("Add: %v", err)
			break
		}
	}
	<-done
}
