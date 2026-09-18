package index

import (
	"strings"
	"unicode"
)

// Tokenize lowercases text and splits it into alphanumeric tokens. Anything
// that is not a letter or a digit acts as a separator, so "GPU-bound (fast!)"
// becomes ["gpu", "bound", "fast"]. The tokenizer is unicode-aware, which
// matters for a corpus that mixes Latin and Cyrillic.
func Tokenize(text string) []string {
	tokens := make([]string, 0, len(text)/6+1)
	var b strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
			continue
		}
		if b.Len() > 0 {
			tokens = append(tokens, b.String())
			b.Reset()
		}
	}
	if b.Len() > 0 {
		tokens = append(tokens, b.String())
	}
	return tokens
}

// charNGrams returns the character n-grams of a single token, padded with a
// boundary marker so that prefixes and suffixes get their own features. The
// padding is what lets "retrieval" and "retrieve" share most of their grams,
// which is where the fuzzy half of the hybrid ranking comes from.
func charNGrams(token string, n int) []string {
	runes := []rune("^" + token + "$")
	if len(runes) <= n {
		return []string{string(runes)}
	}
	grams := make([]string, 0, len(runes)-n+1)
	for i := 0; i+n <= len(runes); i++ {
		grams = append(grams, string(runes[i:i+n]))
	}
	return grams
}
