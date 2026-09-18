package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tosekk/nanoseek/internal/index"
)

func newTestServer(t *testing.T, dataPath string) http.Handler {
	t.Helper()
	ix := index.New()
	for _, d := range []index.Document{
		{ID: "a", Title: "BM25 ranking", Text: "Term frequency with length normalisation."},
		{ID: "b", Title: "Cosine similarity", Text: "The angle between two vectors."},
	} {
		if _, err := ix.Add(d); err != nil {
			t.Fatalf("seed %q: %v", d.ID, err)
		}
	}
	return New(ix, dataPath).Handler()
}

func do(t *testing.T, h http.Handler, method, target, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, target, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	var payload map[string]any
	if ct := rec.Header().Get("Content-Type"); strings.HasPrefix(ct, "application/json") {
		if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
			t.Fatalf("%s %s: decode body %q: %v", method, target, rec.Body.String(), err)
		}
	}
	return rec, payload
}

func TestSearchEndpoint(t *testing.T) {
	h := newTestServer(t, "")
	rec, body := do(t, h, http.MethodGet, "/api/search?q=cosine+vectors&k=1", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if body["mode"] != "hybrid" {
		t.Errorf("mode = %v, want hybrid", body["mode"])
	}
	results, _ := body["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	hit := results[0].(map[string]any)
	doc := hit["document"].(map[string]any)
	if doc["id"] != "b" {
		t.Errorf("top hit = %v, want b", doc["id"])
	}
	if _, ok := hit["score"].(float64); !ok {
		t.Errorf("hit is missing a numeric score: %v", hit)
	}
}

func TestSearchPOSTMatchesGET(t *testing.T) {
	h := newTestServer(t, "")
	_, get := do(t, h, http.MethodGet, "/api/search?q=ranking&mode=lexical", "")
	_, post := do(t, h, http.MethodPost, "/api/search", `{"query":"ranking","mode":"lexical"}`)
	if get["count"] != post["count"] {
		t.Errorf("GET count %v != POST count %v", get["count"], post["count"])
	}
}

func TestSearchValidation(t *testing.T) {
	h := newTestServer(t, "")
	cases := []struct {
		name, target, body string
		method             string
		want               int
	}{
		{"missing query", "/api/search", "", http.MethodGet, http.StatusBadRequest},
		{"bad mode", "/api/search?q=x&mode=magic", "", http.MethodGet, http.StatusBadRequest},
		{"bad k", "/api/search?q=x&k=lots", "", http.MethodGet, http.StatusBadRequest},
		{"bad json", "/api/search", "{oops", http.MethodPost, http.StatusBadRequest},
		{"unknown field", "/api/search", `{"quarry":"typo"}`, http.MethodPost, http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec, body := do(t, h, tc.method, tc.target, tc.body)
			if rec.Code != tc.want {
				t.Fatalf("status = %d, want %d", rec.Code, tc.want)
			}
			if body["error"] == nil {
				t.Errorf("response has no error message: %v", body)
			}
		})
	}
}

func TestAddAndDeleteDocumentPersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	h := newTestServer(t, path)

	rec, body := do(t, h, http.MethodPost, "/api/documents", `{"title":"Kayaks","text":"Paddling a narrow boat."}`)
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", rec.Code)
	}
	id, _ := body["id"].(string)
	if id == "" {
		t.Fatal("response carried no id")
	}

	saved, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("corpus was not written: %v", err)
	}
	if !strings.Contains(string(saved), "Kayaks") {
		t.Errorf("saved corpus is missing the new document:\n%s", saved)
	}

	if _, found := do(t, h, http.MethodGet, "/api/search?q=paddling&mode=lexical", ""); found["count"].(float64) != 1 {
		t.Errorf("new document is not searchable: %v", found)
	}

	rec, _ = do(t, h, http.MethodDelete, "/api/documents/"+id, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200", rec.Code)
	}
	if rec, _ = do(t, h, http.MethodDelete, "/api/documents/"+id, ""); rec.Code != http.StatusNotFound {
		t.Errorf("second delete status = %d, want 404", rec.Code)
	}

	saved, err = os.ReadFile(path)
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	if strings.Contains(string(saved), "Kayaks") {
		t.Errorf("deleted document is still on disk:\n%s", saved)
	}
}

func TestStatsAndDocumentListing(t *testing.T) {
	h := newTestServer(t, "")
	_, stats := do(t, h, http.MethodGet, "/api/stats", "")
	if stats["documents"].(float64) != 2 {
		t.Errorf("documents = %v, want 2", stats["documents"])
	}
	_, list := do(t, h, http.MethodGet, "/api/documents", "")
	if list["count"].(float64) != 2 {
		t.Errorf("count = %v, want 2", list["count"])
	}
}

func TestPlaygroundIsServedAtRoot(t *testing.T) {
	h := newTestServer(t, "")
	rec, _ := do(t, h, http.MethodGet, "/", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "<title>nanoseek</title>") {
		t.Error("root did not serve the playground HTML")
	}
}

func TestWrongMethodIsRejected(t *testing.T) {
	h := newTestServer(t, "")
	rec, _ := do(t, h, http.MethodPost, "/api/stats", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}
