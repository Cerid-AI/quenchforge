// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package registry

// CatalogEntry is one well-tested (alias, repo, file-match) tuple.
// Operators say `quenchforge pull llama3.2:3b` and the resolver looks
// up the alias here.
type CatalogEntry struct {
	Alias     string
	Repo      string
	FileMatch string // substring to match inside the repo's GGUF filenames
	LocalName string // on-disk basename (without .gguf)
	VRAMGB    int    // approx VRAM footprint of a single instance
	Notes     string
}

// catalog is curated against the AMD-Mac VRAM tiers documented in
// cerid-ai's `docs/AMD_GPU_MODEL_RECOMMENDATIONS.md`. Picks are biased
// toward Q4_K_M quants (best speed/quality tradeoff on AMD Mac scalar
// fallback) and the bartowski / IlyaGusev imatrix repos (well-known
// for calibration quality).
//
// Keeping the catalog small and well-tested is a feature — operators
// looking for exotic quants should pass the full `<repo>:<quant>` spec
// instead. This catalog is the "did you mean..." landing pad for new
// users, not an exhaustive index of GGUFs on HF.
var catalog = []CatalogEntry{
	// ---- Chat / completion ----
	{
		Alias:     "llama3.2:3b",
		Repo:      "bartowski/Llama-3.2-3B-Instruct-GGUF",
		FileMatch: "Q4_K_M",
		LocalName: "llama-3.2-3b-instruct-q4_k_m",
		VRAMGB:    3,
		Notes:     "Small + fast — fits comfortably on 8 GB tier (Apple Silicon Air, Vega Pro 16 GB)",
	},
	{
		Alias:     "llama3.1:8b",
		Repo:      "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
		FileMatch: "Q4_K_M",
		LocalName: "meta-llama-3.1-8b-instruct-q4_k_m",
		VRAMGB:    6,
		Notes:     "Balanced — 8 GB tier and up",
	},
	{
		Alias:     "qwen2.5:7b",
		Repo:      "bartowski/Qwen2.5-7B-Instruct-GGUF",
		FileMatch: "Q4_K_M",
		LocalName: "qwen2.5-7b-instruct-q4_k_m",
		VRAMGB:    5,
		Notes:     "Strong reasoning at 7B — 8 GB tier and up",
	},
	{
		Alias:     "qwen2.5:14b",
		Repo:      "bartowski/Qwen2.5-14B-Instruct-GGUF",
		FileMatch: "Q4_K_M",
		LocalName: "qwen2.5-14b-instruct-q4_k_m",
		VRAMGB:    10,
		Notes:     "Higher quality — 16 GB tier minimum",
	},
	{
		Alias:     "qwen2.5:32b",
		Repo:      "bartowski/Qwen2.5-32B-Instruct-GGUF",
		FileMatch: "Q4_K_M",
		LocalName: "qwen2.5-32b-instruct-q4_k_m",
		VRAMGB:    21,
		Notes:     "Top quality at this size — 32 GB tier (Vega II Duo, W6900X)",
	},

	// ---- Dense embeddings ----
	{
		Alias:     "nomic-embed:v1.5",
		Repo:      "nomic-ai/nomic-embed-text-v1.5-GGUF",
		FileMatch: "Q8_0",
		LocalName: "nomic-embed-text-v1.5",
		VRAMGB:    1,
		Notes:     "768-dim dense embedder, the standard cerid-ai recommendation",
	},
	{
		Alias:     "bge-m3",
		Repo:      "second-state/bge-m3-GGUF",
		FileMatch: "f16",
		LocalName: "bge-m3-f16",
		VRAMGB:    2,
		Notes:     "Multilingual + multi-functional (dense, sparse, ColBERT) — larger than nomic",
	},

	// ---- Cross-encoder rerankers ----
	{
		Alias:     "bge-reranker:v2-m3",
		Repo:      "gpustack/bge-reranker-v2-m3-GGUF",
		FileMatch: "Q4_K_M",
		LocalName: "bge-reranker-v2-m3",
		VRAMGB:    1,
		Notes:     "Multilingual reranker, fast and accurate",
	},
}

// lookupAlias returns the catalog entry matching the alias case-insensitively,
// or false if not in the catalog.
func lookupAlias(alias string) (CatalogEntry, bool) {
	for _, e := range catalog {
		if equalIgnoreCase(e.Alias, alias) {
			return e, true
		}
	}
	return CatalogEntry{}, false
}

// Catalog returns the full catalog — used by `quenchforge pull --list`
// to show available aliases.
func Catalog() []CatalogEntry {
	out := make([]CatalogEntry, len(catalog))
	copy(out, catalog)
	return out
}

// equalIgnoreCase is a small ASCII-only case-insensitive compare to
// avoid pulling in `strings.EqualFold` (which has Unicode-folding
// surprises around the dotless-I — irrelevant here, but the explicit
// behavior is clearer in code).
func equalIgnoreCase(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 32
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 32
		}
		if ca != cb {
			return false
		}
	}
	return true
}
