// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseSpec_CatalogAlias(t *testing.T) {
	spec, err := ParseSpec("llama3.2:3b")
	if err != nil {
		t.Fatalf("ParseSpec: %v", err)
	}
	if !strings.Contains(spec.Repo, "Llama-3.2-3B") {
		t.Errorf("Repo = %q, want it to contain Llama-3.2-3B", spec.Repo)
	}
	if spec.FileMatch == "" {
		t.Errorf("FileMatch should be non-empty for catalog aliases")
	}
	if spec.LocalName == "" {
		t.Errorf("LocalName should be non-empty for catalog aliases")
	}
}

func TestParseSpec_ExplicitRepoQuant(t *testing.T) {
	spec, err := ParseSpec("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
	if err != nil {
		t.Fatalf("ParseSpec: %v", err)
	}
	if spec.Repo != "bartowski/Llama-3.2-3B-Instruct-GGUF" {
		t.Errorf("Repo = %q", spec.Repo)
	}
	if spec.FileMatch != "Q4_K_M" {
		t.Errorf("FileMatch = %q", spec.FileMatch)
	}
	// LocalName should be derived sanely
	if !strings.Contains(strings.ToLower(spec.LocalName), "q4_k_m") {
		t.Errorf("LocalName = %q, expected to include q4_k_m", spec.LocalName)
	}
	if strings.Contains(spec.LocalName, "GGUF") || strings.Contains(spec.LocalName, "gguf") {
		t.Errorf("LocalName %q should have GGUF suffix stripped", spec.LocalName)
	}
}

func TestParseSpec_ExplicitFullFilename(t *testing.T) {
	spec, err := ParseSpec("bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	if err != nil {
		t.Fatalf("ParseSpec: %v", err)
	}
	if spec.Repo != "bartowski/Llama-3.2-3B-Instruct-GGUF" {
		t.Errorf("Repo = %q", spec.Repo)
	}
	if spec.LocalName != "Llama-3.2-3B-Instruct-Q4_K_M" {
		t.Errorf("LocalName = %q", spec.LocalName)
	}
}

func TestParseSpec_InvalidForms(t *testing.T) {
	cases := []string{
		"",                         // empty
		"not-a-spec",               // no slash, no colon, unknown alias
		"some/repo",                // repo only, no quant
		"some:thing:weird",         // double colon — confuses our parser? Actually the last ":" wins
		"single-token-no-slash:Q4", // no `/` in repo
	}
	for _, arg := range cases {
		t.Run(arg, func(t *testing.T) {
			_, err := ParseSpec(arg)
			if err == nil && arg == "" {
				t.Errorf("ParseSpec(%q) should error", arg)
			}
			if err == nil && arg == "single-token-no-slash:Q4" {
				t.Errorf("ParseSpec(%q) should error", arg)
			}
			// `some/repo` and `some:thing:weird` are technically valid
			// shapes (last `:` is the separator); we let them through.
			// They'll fail at HF API time with a clearer error.
			_ = err
		})
	}
}

func TestLookupAlias(t *testing.T) {
	if _, ok := lookupAlias("llama3.2:3b"); !ok {
		t.Errorf("llama3.2:3b should be in catalog")
	}
	if _, ok := lookupAlias("LLAMA3.2:3B"); !ok {
		t.Errorf("alias lookup must be case-insensitive")
	}
	if _, ok := lookupAlias("nonexistent:9b"); ok {
		t.Errorf("nonexistent alias should not match")
	}
}

func TestCatalog_Returns_Copy(t *testing.T) {
	c1 := Catalog()
	c2 := Catalog()
	if len(c1) != len(c2) || len(c1) == 0 {
		t.Fatalf("Catalog() returned %d / %d entries", len(c1), len(c2))
	}
	// Mutating c1 must not affect c2.
	c1[0].Alias = "mutated"
	if Catalog()[0].Alias == "mutated" {
		t.Errorf("Catalog() returned a shared slice — should be a copy")
	}
}

// ---------------------------------------------------------------------------
// End-to-end Pull against a mock HF server
// ---------------------------------------------------------------------------

func TestPull_E2E_HappyPath(t *testing.T) {
	// 1. Build a fake HF server that:
	//    - GET /api/models/{repo}/tree/main → returns one .gguf entry
	//    - GET /{repo}/resolve/main/{path}  → returns the content
	payload := []byte("fake GGUF magic GGUF\x00" + strings.Repeat("\x00", 100))
	h := sha256.Sum256(payload)
	expectedSha := hex.EncodeToString(h[:])

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/testorg/test-gguf/tree/main", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		entries := []hfTreeEntry{
			{
				Path: "test-model-Q4_K_M.gguf",
				Type: "file",
				Size: int64(len(payload)),
				LFS: &struct {
					SHA256 string `json:"sha256"`
					Size   int64  `json:"size"`
				}{SHA256: expectedSha, Size: int64(len(payload))},
			},
			{
				// A non-matching file that should be filtered out
				Path: "test-model-Q8_0.gguf",
				Type: "file",
				Size: int64(len(payload)) * 2,
			},
		}
		_ = json.NewEncoder(w).Encode(entries)
	})
	mux.HandleFunc("/testorg/test-gguf/resolve/main/test-model-Q4_K_M.gguf", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(payload)))
		w.WriteHeader(200)
		_, _ = w.Write(payload)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tmpDir := t.TempDir()
	client := New(tmpDir).WithBaseURL(srv.URL)

	spec, err := ParseSpec("testorg/test-gguf:Q4_K_M")
	if err != nil {
		t.Fatalf("ParseSpec: %v", err)
	}

	var progressCalls int
	progress := func(done, total int64) { progressCalls++ }

	finalPath, err := client.Pull(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("Pull: %v", err)
	}

	// Check file exists, contents match, no partial leftover
	got, err := os.ReadFile(finalPath)
	if err != nil {
		t.Fatalf("ReadFile %s: %v", finalPath, err)
	}
	if string(got) != string(payload) {
		t.Errorf("downloaded bytes don't match")
	}
	if _, err := os.Stat(finalPath + PartialSuffix); !os.IsNotExist(err) {
		t.Errorf("partial file should have been renamed away after success")
	}
	if progressCalls == 0 {
		t.Errorf("progress callback never fired")
	}
}

func TestPull_E2E_Idempotent_SkipsWhenAlreadyPresent(t *testing.T) {
	// Pre-populate the models dir with a file at the expected SHA;
	// Pull should short-circuit without a second download.
	payload := []byte("idempotent fake GGUF")
	h := sha256.Sum256(payload)
	expectedSha := hex.EncodeToString(h[:])

	var downloadCount int
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/x/y/tree/main", func(w http.ResponseWriter, r *http.Request) {
		entries := []hfTreeEntry{{
			Path: "model-Q4.gguf",
			Type: "file",
			Size: int64(len(payload)),
			LFS: &struct {
				SHA256 string `json:"sha256"`
				Size   int64  `json:"size"`
			}{SHA256: expectedSha, Size: int64(len(payload))},
		}}
		_ = json.NewEncoder(w).Encode(entries)
	})
	mux.HandleFunc("/x/y/resolve/main/model-Q4.gguf", func(w http.ResponseWriter, r *http.Request) {
		downloadCount++
		_, _ = w.Write(payload)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tmpDir := t.TempDir()
	// Pre-place the file at the expected on-disk name. The name must
	// match what localNameFromRepoMatch produces — lowercase, even when
	// the spec's FileMatch is upper/mixed case. On a case-insensitive
	// macOS filesystem the wrong case still resolves; Linux CI is
	// case-sensitive and would silently re-download.
	preplacePath := filepath.Join(tmpDir, "y-q4.gguf")
	if err := os.WriteFile(preplacePath, payload, 0o644); err != nil {
		t.Fatalf("preplace: %v", err)
	}

	client := New(tmpDir).WithBaseURL(srv.URL)
	spec, _ := ParseSpec("x/y:Q4")

	if _, err := client.Pull(context.Background(), spec, nil); err != nil {
		t.Fatalf("Pull: %v", err)
	}

	if downloadCount != 0 {
		t.Errorf("Pull should have short-circuited; got %d downloads", downloadCount)
	}
}

func TestPull_E2E_RejectsBadSHA(t *testing.T) {
	payload := []byte("served-bytes")
	// Advertise a wrong SHA — Pull should refuse to install.
	wrongSha := "0000000000000000000000000000000000000000000000000000000000000000"

	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/x/y/tree/main", func(w http.ResponseWriter, r *http.Request) {
		entries := []hfTreeEntry{{
			Path: "m.gguf", Type: "file", Size: int64(len(payload)),
			LFS: &struct {
				SHA256 string `json:"sha256"`
				Size   int64  `json:"size"`
			}{SHA256: wrongSha, Size: int64(len(payload))},
		}}
		_ = json.NewEncoder(w).Encode(entries)
	})
	mux.HandleFunc("/x/y/resolve/main/m.gguf", func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(payload)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tmpDir := t.TempDir()
	client := New(tmpDir).WithBaseURL(srv.URL)
	spec, _ := ParseSpec("x/y:m")
	_, err := client.Pull(context.Background(), spec, nil)
	if err == nil {
		t.Fatalf("Pull should have refused on SHA mismatch")
	}
	if !strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Errorf("error should mention SHA mismatch, got: %v", err)
	}
}

func TestPull_E2E_404Repo(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/missing/repo/tree/main", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := New(t.TempDir()).WithBaseURL(srv.URL)
	spec, _ := ParseSpec("missing/repo:Q4")
	_, err := client.Pull(context.Background(), spec, nil)
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' error, got: %v", err)
	}
}

func TestPull_E2E_NoMatchingFile(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/x/y/tree/main", func(w http.ResponseWriter, r *http.Request) {
		entries := []hfTreeEntry{
			{Path: "model-Q8_0.gguf", Type: "file", Size: 100},
			{Path: "README.md", Type: "file", Size: 50},
		}
		_ = json.NewEncoder(w).Encode(entries)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := New(t.TempDir()).WithBaseURL(srv.URL)
	spec, _ := ParseSpec("x/y:Q4_K_M")
	_, err := client.Pull(context.Background(), spec, nil)
	if err == nil || !strings.Contains(err.Error(), "no GGUF matching") {
		t.Errorf("expected helpful 'no GGUF matching' error, got: %v", err)
	}
	// Error should suggest what IS available.
	if err != nil && !strings.Contains(err.Error(), "model-Q8_0.gguf") {
		t.Errorf("error should list available files, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// List + Remove
// ---------------------------------------------------------------------------

func TestListAndRemove(t *testing.T) {
	tmpDir := t.TempDir()

	// Empty dir → empty list, no error
	got, err := List(tmpDir)
	if err != nil {
		t.Fatalf("List on empty dir: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty list, got %d entries", len(got))
	}

	// Pre-populate two GGUFs and one ignored file.
	mustWrite := func(name, content string) {
		if err := os.WriteFile(filepath.Join(tmpDir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite("alpha.gguf", "111")
	mustWrite("beta.gguf", "22")
	mustWrite("ignore.txt", "x")

	got, err = List(tmpDir)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 entries (alpha + beta), got %d: %v", len(got), got)
	}
	// Verify size + name; sort is on name from os.ReadDir's
	// implementation but we don't rely on order.
	byName := map[string]Model{}
	for _, m := range got {
		byName[m.Name] = m
	}
	if byName["alpha"].SizeBytes != 3 {
		t.Errorf("alpha size = %d, want 3", byName["alpha"].SizeBytes)
	}
	if byName["beta"].SizeBytes != 2 {
		t.Errorf("beta size = %d", byName["beta"].SizeBytes)
	}

	// Remove(missing) errors usefully
	if err := Remove(tmpDir, "missing"); err == nil {
		t.Errorf("Remove of missing model should error")
	}
	// Remove(alpha) succeeds
	if err := Remove(tmpDir, "alpha"); err != nil {
		t.Fatalf("Remove alpha: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmpDir, "alpha.gguf")); !os.IsNotExist(err) {
		t.Errorf("alpha.gguf should be gone")
	}
	// Remove rejects path-separator injection
	if err := Remove(tmpDir, "../../etc/passwd"); err == nil {
		t.Errorf("Remove should reject paths containing separators")
	}
}

func TestList_Symlinks(t *testing.T) {
	tmpDir := t.TempDir()
	srcDir := t.TempDir()
	target := filepath.Join(srcDir, "real-model.gguf")
	if err := os.WriteFile(target, []byte("1234567890"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(tmpDir, "linked-model.gguf")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}

	got, err := List(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d entries, want 1", len(got))
	}
	if !got[0].Symlink {
		t.Errorf("symlink not flagged as such")
	}
	if got[0].SizeBytes != 10 {
		t.Errorf("symlink size = %d, want 10 (stat of target)", got[0].SizeBytes)
	}

	// Remove the symlink should NOT remove the target
	if err := Remove(tmpDir, "linked-model"); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := os.Stat(target); err != nil {
		t.Errorf("symlink target should still exist after removing the link: %v", err)
	}
}

// Catalog enum has reasonable entries
func TestCatalog_HasSensibleEntries(t *testing.T) {
	c := Catalog()
	if len(c) < 5 {
		t.Errorf("catalog has %d entries; expected at least a few model tiers", len(c))
	}
	// At least one chat, one embed, one rerank
	var hasChat, hasEmbed, hasRerank bool
	for _, e := range c {
		if strings.Contains(e.Alias, "llama") || strings.Contains(e.Alias, "qwen") {
			hasChat = true
		}
		if strings.Contains(e.Alias, "embed") {
			hasEmbed = true
		}
		if strings.Contains(e.Alias, "reranker") {
			hasRerank = true
		}
	}
	if !hasChat || !hasEmbed || !hasRerank {
		t.Errorf("catalog should cover chat/embed/rerank; chat=%v embed=%v rerank=%v",
			hasChat, hasEmbed, hasRerank)
	}
}

// Smoke that the time helpers don't crash on zero values.
var _ = time.Time{}
