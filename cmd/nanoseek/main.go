// Command nanoseek is a tiny hybrid search engine: a BM25 lexical ranker and a
// hashed-feature vector ranker fused with reciprocal rank fusion, wrapped in a
// CLI and an HTTP server with a browser playground.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/sciddles/nanoseek/internal/index"
	"github.com/sciddles/nanoseek/internal/server"
)

const defaultData = "data/corpus.jsonl"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "nanoseek:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return errors.New("no command given")
	}
	switch args[0] {
	case "serve":
		return serve(args[1:])
	case "query":
		return query(args[1:])
	case "add":
		return add(args[1:])
	case "stats":
		return stats(args[1:])
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		usage()
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `nanoseek — hybrid search in a single binary

  nanoseek serve [-addr :8080] [-data data/corpus.jsonl]
      Load the corpus and serve the API plus the browser playground.

  nanoseek query "<text>" [-k 5] [-mode hybrid|lexical|vector] [-data ...]
      Search the corpus from the terminal.

  nanoseek add -title "..." -text "..." [-data ...]
      Append one document to the corpus and reindex it.

  nanoseek stats [-data ...]
      Print index size.

`)
}

// splitArgs separates flags from positional words so that both
// `query -mode vector "text"` and `query "text" -mode vector` work. Go's flag
// package stops parsing at the first positional argument, which makes the
// second form silently drop every flag after it.
func splitArgs(args []string, valueFlags ...string) (flags, positional []string) {
	takesValue := make(map[string]bool, len(valueFlags))
	for _, name := range valueFlags {
		takesValue[name] = true
	}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			positional = append(positional, arg)
			continue
		}
		flags = append(flags, arg)
		name := strings.TrimLeft(arg, "-")
		// A flag written as -k 3 swallows the next argument; -k=3 does not.
		if !strings.Contains(arg, "=") && takesValue[name] && i+1 < len(args) {
			i++
			flags = append(flags, args[i])
		}
	}
	return flags, positional
}

// load builds an index from a JSONL corpus on disk.
func load(path string) (*index.Index, error) {
	ix := index.New()
	n, err := ix.LoadJSONL(path)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", path, err)
	}
	fmt.Fprintf(os.Stderr, "indexed %d documents from %s\n", n, path)
	return ix, nil
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "address to listen on")
	data := fs.String("data", defaultData, "JSONL corpus to load and persist to")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ix, err := load(*data)
	if err != nil {
		return err
	}

	srv := &http.Server{
		Addr:              *addr,
		Handler:           server.New(ix, *data).Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	// Shut down on Ctrl-C so an in-flight save is never cut in half.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 1)
	go func() {
		fmt.Fprintf(os.Stderr, "playground on http://localhost%s\n", *addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errs <- err
		}
	}()

	select {
	case err := <-errs:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		fmt.Fprintln(os.Stderr, "\nshutting down")
		return srv.Shutdown(shutdownCtx)
	}
}

func query(args []string) error {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	data := fs.String("data", defaultData, "JSONL corpus to search")
	k := fs.Int("k", 5, "number of results")
	modeFlag := fs.String("mode", "hybrid", "hybrid, lexical or vector")
	flags, words := splitArgs(args, "data", "k", "mode")
	if err := fs.Parse(flags); err != nil {
		return err
	}
	if len(words) == 0 {
		return errors.New("query: give me something to search for")
	}
	mode, err := index.ParseMode(*modeFlag)
	if err != nil {
		return err
	}

	ix, err := load(*data)
	if err != nil {
		return err
	}

	results := ix.Search(strings.Join(words, " "), *k, mode)
	if len(results) == 0 {
		fmt.Println("no matches")
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "#\tSCORE\tBM25\tCOSINE\tTITLE")
	for i, r := range results {
		title := r.Document.Title
		if title == "" {
			title = r.Document.ID
		}
		fmt.Fprintf(w, "%d\t%.4f\t%.3f\t%.3f\t%s\n", i+1, r.Score, r.Lexical, r.Vector, title)
	}
	if err := w.Flush(); err != nil {
		return err
	}
	fmt.Println()
	for i, r := range results {
		fmt.Printf("%d. %s\n   %s\n\n", i+1, r.Document.ID, r.Snippet)
	}
	return nil
}

func add(args []string) error {
	fs := flag.NewFlagSet("add", flag.ExitOnError)
	data := fs.String("data", defaultData, "JSONL corpus to append to")
	id := fs.String("id", "", "document id (generated when empty)")
	title := fs.String("title", "", "document title")
	text := fs.String("text", "", "document text")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ix, err := load(*data)
	if err != nil {
		return err
	}
	newID, err := ix.Add(index.Document{ID: *id, Title: *title, Text: *text})
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*data), 0o755); err != nil {
		return err
	}
	if err := ix.SaveJSONL(*data); err != nil {
		return err
	}
	fmt.Println("added", newID)
	return nil
}

func stats(args []string) error {
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	data := fs.String("data", defaultData, "JSONL corpus to inspect")
	if err := fs.Parse(args); err != nil {
		return err
	}
	ix, err := load(*data)
	if err != nil {
		return err
	}
	s := ix.Stats()
	fmt.Printf("documents: %d\nterms:     %d\navg length: %.1f tokens\n",
		s.Documents, s.Terms, s.AvgLength)
	return nil
}
