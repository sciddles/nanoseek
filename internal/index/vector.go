package index

import (
	"hash/fnv"
	"math"
)

// vectorDim is the size of the hashing-trick feature space. Features are FNV
// hashes of tokens and character n-grams folded into this many buckets, so the
// vocabulary can grow without the vectors growing with it.
const vectorDim = 1 << 15

// ngramSize is the character n-gram width used for the fuzzy features.
const ngramSize = 3

// gramWeight is how much a character n-gram counts relative to a whole word.
// Words carry the signal; grams are there to survive typos and inflection.
const gramWeight = 0.35

// Vector is a sparse, L2-normalized bag of hashed features. It is stored as a
// map because a document touches only a few hundred of the 32k buckets.
type Vector map[uint32]float64

// Embed turns text into a unit-length sparse vector. No model, no network
// call: the "embedding" is a deterministic hash of word and character features.
// It captures lexical similarity only, which is exactly what the vector half of
// the hybrid score is asked to do here. Swapping this for a real embedding API
// means replacing this one function.
func Embed(text string) Vector {
	words := make(map[uint32]float64)
	grams := make(map[uint32]float64)
	for _, token := range Tokenize(text) {
		words[bucket(token)]++
		for _, gram := range charNGrams(token, ngramSize) {
			grams[bucket(gram)]++
		}
	}

	v := make(Vector, len(words)+len(grams))
	// Sublinear scaling keeps a word repeated twenty times from drowning out
	// the rest of the document, the same reason BM25 saturates term frequency.
	// It is applied to the raw count before weighting, so the weight never
	// pushes a feature below zero.
	for k, count := range words {
		v[k] += 1 + math.Log(count)
	}
	for k, count := range grams {
		v[k] += gramWeight * (1 + math.Log(count))
	}

	var norm float64
	for _, w := range v {
		norm += w * w
	}
	if norm == 0 {
		return v
	}
	norm = math.Sqrt(norm)
	for k := range v {
		v[k] /= norm
	}
	return v
}

// Cosine is the cosine similarity of two vectors from Embed. Both are already
// unit-length, so this is just their dot product; it iterates the smaller
// vector to keep the cost proportional to the query rather than the document.
func Cosine(a, b Vector) float64 {
	if len(a) > len(b) {
		a, b = b, a
	}
	var dot float64
	for k, av := range a {
		if bv, ok := b[k]; ok {
			dot += av * bv
		}
	}
	return dot
}

func bucket(s string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s))
	return h.Sum32() % vectorDim
}
