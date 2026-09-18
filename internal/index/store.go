package index

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// LoadJSONL reads documents from a JSON Lines file and indexes them. A missing
// file is not an error: a fresh corpus simply starts empty.
func (ix *Index) LoadJSONL(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return 0, nil
		}
		return 0, err
	}
	defer f.Close()
	return ix.ReadJSONL(f)
}

// ReadJSONL indexes documents from a JSON Lines stream, one object per line.
// Blank lines are skipped; a malformed line stops the load and names itself.
func (ix *Index) ReadJSONL(r io.Reader) (int, error) {
	scanner := bufio.NewScanner(r)
	// Corpus entries can be long prose, so allow lines well past bufio's 64KB.
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)

	count := 0
	for line := 1; scanner.Scan(); line++ {
		raw := strings.TrimSpace(scanner.Text())
		if raw == "" {
			continue
		}
		var doc Document
		if err := json.Unmarshal([]byte(raw), &doc); err != nil {
			return count, fmt.Errorf("line %d: %w", line, err)
		}
		if _, err := ix.Add(doc); err != nil {
			return count, fmt.Errorf("line %d: %w", line, err)
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		return count, err
	}
	return count, nil
}

// SaveJSONL writes every live document back out as JSON Lines. It writes to a
// temporary file in the same directory and renames it into place, so an
// interrupted save cannot leave a half-written corpus behind.
func (ix *Index) SaveJSONL(path string) error {
	docs := ix.Documents()

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	w := bufio.NewWriter(tmp)
	enc := json.NewEncoder(w)
	for _, doc := range docs {
		if err := enc.Encode(doc); err != nil {
			tmp.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
