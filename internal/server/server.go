// Package server exposes the search index over HTTP and serves the browser
// playground that drives it.
package server

import (
	"embed"
	"encoding/json"
	"io/fs"
	"log"
	"net/http"
	"strconv"
	"sync"

	"github.com/sciddles/nanoseek/internal/index"
)

//go:embed static
var staticFS embed.FS

// maxBodyBytes caps request bodies. Documents are prose, not uploads.
const maxBodyBytes = 4 << 20

// Server wires the index to HTTP handlers and, when a data path is configured,
// persists the corpus after every change.
type Server struct {
	ix       *index.Index
	dataPath string

	saveMu sync.Mutex
}

// New returns a Server over ix. If dataPath is empty the index stays in memory
// only and changes are lost when the process exits.
func New(ix *index.Index, dataPath string) *Server {
	return &Server{ix: ix, dataPath: dataPath}
}

// Handler builds the router. Patterns use Go's method-aware mux, so a wrong
// verb answers 405 without any routing code of our own.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("GET /api/documents", s.handleListDocuments)
	mux.HandleFunc("POST /api/documents", s.handleAddDocument)
	mux.HandleFunc("DELETE /api/documents/{id}", s.handleDeleteDocument)
	mux.HandleFunc("GET /api/search", s.handleSearchGET)
	mux.HandleFunc("POST /api/search", s.handleSearchPOST)

	ui, err := fs.Sub(staticFS, "static")
	if err != nil {
		// The directory is embedded at build time, so this cannot fail at run
		// time; panicking beats serving a playground-less server silently.
		panic(err)
	}
	mux.Handle("GET /", http.FileServerFS(ui))

	return logRequests(mux)
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.ix.Stats())
}

func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	docs := s.ix.Documents()
	writeJSON(w, http.StatusOK, map[string]any{
		"count":     len(docs),
		"documents": docs,
	})
}

func (s *Server) handleAddDocument(w http.ResponseWriter, r *http.Request) {
	var doc index.Document
	if !decodeJSON(w, r, &doc) {
		return
	}
	id, err := s.ix.Add(doc)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.persist()
	writeJSON(w, http.StatusCreated, map[string]any{"id": id})
}

func (s *Server) handleDeleteDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.ix.Delete(id) {
		writeError(w, http.StatusNotFound, "no document with id "+id)
		return
	}
	s.persist()
	writeJSON(w, http.StatusOK, map[string]any{"deleted": id})
}

// searchRequest is the POST body for /api/search. K and Mode are optional.
type searchRequest struct {
	Query string `json:"query"`
	K     int    `json:"k"`
	Mode  string `json:"mode"`
}

func (s *Server) handleSearchPOST(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	s.search(w, req)
}

func (s *Server) handleSearchGET(w http.ResponseWriter, r *http.Request) {
	req := searchRequest{
		Query: r.URL.Query().Get("q"),
		Mode:  r.URL.Query().Get("mode"),
	}
	if raw := r.URL.Query().Get("k"); raw != "" {
		k, err := strconv.Atoi(raw)
		if err != nil {
			writeError(w, http.StatusBadRequest, "k must be a number")
			return
		}
		req.K = k
	}
	s.search(w, req)
}

func (s *Server) search(w http.ResponseWriter, req searchRequest) {
	if req.Query == "" {
		writeError(w, http.StatusBadRequest, "query is required")
		return
	}
	mode, err := index.ParseMode(req.Mode)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.K == 0 {
		req.K = 10
	}
	results := s.ix.Search(req.Query, req.K, mode)
	writeJSON(w, http.StatusOK, map[string]any{
		"query":   req.Query,
		"mode":    string(mode),
		"count":   len(results),
		"results": results,
	})
}

// persist writes the corpus to disk after a change. A failed save is logged
// rather than returned: the write to the index itself already succeeded, and
// failing the request would wrongly suggest it did not.
func (s *Server) persist() {
	if s.dataPath == "" {
		return
	}
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	if err := s.ix.SaveJSONL(s.dataPath); err != nil {
		log.Printf("save %s: %v", s.dataPath, err)
	}
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r)
		log.Printf("%s %s", r.Method, r.URL.RequestURI())
	})
}
