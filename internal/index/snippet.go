package index

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// snippetWidth is the length in runes of the excerpt returned with each hit.
const snippetWidth = 240

// snippet returns a window of text centred on the first query term that
// appears in it, so a result shows the matching passage rather than whatever
// happens to be at the top of the document.
func snippet(text, query string) string {
	text = strings.Join(strings.Fields(text), " ")
	runes := []rune(text)
	if len(runes) <= snippetWidth {
		return text
	}

	hit := firstMatch(text, query)
	if hit < 0 {
		return string(runes[:snippetWidth]) + "…"
	}

	start := hit - snippetWidth/3
	if start < 0 {
		start = 0
	}
	end := start + snippetWidth
	if end > len(runes) {
		end = len(runes)
		start = end - snippetWidth
	}

	// Trim to whole words so the excerpt does not begin or end mid-token.
	for start > 0 && !unicode.IsSpace(runes[start-1]) {
		start++
	}
	for end < len(runes) && !unicode.IsSpace(runes[end]) {
		end--
	}

	out := strings.TrimSpace(string(runes[start:end]))
	if start > 0 {
		out = "…" + out
	}
	if end < len(runes) {
		out += "…"
	}
	return out
}

// firstMatch returns the rune offset of the earliest occurrence of any query
// term, or -1 if none of them appear.
func firstMatch(text, query string) int {
	lower := strings.ToLower(text)
	best := -1
	for _, term := range Tokenize(query) {
		if at := strings.Index(lower, term); at >= 0 && (best < 0 || at < best) {
			best = at
		}
	}
	if best < 0 {
		return -1
	}
	return utf8.RuneCountInString(text[:best])
}
