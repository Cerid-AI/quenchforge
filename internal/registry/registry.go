// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package registry implements GGUF model management for Quenchforge:
// pull from HuggingFace, list installed, remove. The biggest UX gap vs
// Ollama is the model on-ramp — pre-v0.4, operators had to manually
// place GGUF files in ~/.quenchforge/models/ or symlink them from an
// existing Ollama install. This package gives them `quenchforge pull
// bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` UX.
//
// Design notes:
//
//   - HF Hub API used directly (no SDK dependency). Two endpoints
//     suffice: GET /api/models/{repo}/tree/main for file listing,
//     and GET /{repo}/resolve/main/{filename} for download.
//   - Atomic writes: download to a tmpfile with a `.qf-partial` suffix,
//     fsync, rename to the final path on success. Partial downloads
//     get cleaned up on failure or resumed via HTTP Range on retry.
//   - ETag integrity: compare the SHA256 from HF's API to a local
//     SHA256 of the downloaded bytes. Refuse to install on mismatch.
//   - Progress reporting via an opt-in callback so the CLI can render
//     a progress bar without this package depending on a TTY library.
//   - Friendly aliases: a small embedded catalog of (alias, repo, file)
//     tuples for the AMD-Mac VRAM tiers documented in
//     `docs/AMD_GPU_MODEL_RECOMMENDATIONS.md`. Operators can resolve
//     `quenchforge pull llama3.2:3b` without knowing the HF repo. The
//     catalog is in `catalog.go`; this file just consumes it.
//
// Out of scope for v0.4:
//   - Quantization (use `llama.cpp/quantize` directly)
//   - Multi-file GGUF (split-archive models — none of our recommended
//     ones use that format yet)
//   - Auth tokens for private HF repos (env: HF_TOKEN supported but
//     gated repos remain a manual operator task)

package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Default HF API base. Tests override via WithBaseURL.
const defaultHFBase = "https://huggingface.co"

// PartialSuffix is the suffix appended to in-progress downloads.
// Exported so callers can clean up stray partials at startup.
const PartialSuffix = ".qf-partial"

// Model is one cached GGUF on disk. Returned by List.
type Model struct {
	// Name is the bare filename without `.gguf` (e.g., "llama3.2-3b").
	Name string

	// Path is the full filesystem path.
	Path string

	// SizeBytes is the file size in bytes (0 for broken symlinks).
	SizeBytes int64

	// ModifiedAt is the file's mtime.
	ModifiedAt time.Time

	// Symlink is true when the entry is a symlink (e.g., something
	// `quenchforge migrate-from-ollama` created).
	Symlink bool
}

// ProgressFn is called periodically during download with (bytes done,
// bytes total). Total is -1 when unknown.
type ProgressFn func(done, total int64)

// Client talks to HuggingFace and writes GGUFs to ModelsDir.
type Client struct {
	httpClient *http.Client
	baseURL    string
	hfToken    string // HF_TOKEN for private/gated repos; empty = anonymous
	modelsDir  string
}

// New returns a Client. modelsDir is the destination for pulls; it's
// created on first write. baseURL defaults to the HF Hub; tests pass an
// httptest server URL.
func New(modelsDir string) *Client {
	return &Client{
		httpClient: &http.Client{Timeout: 0}, // no timeout — multi-GB downloads
		baseURL:    defaultHFBase,
		hfToken:    os.Getenv("HF_TOKEN"),
		modelsDir:  modelsDir,
	}
}

// WithBaseURL overrides the HF API base — test injection only.
func (c *Client) WithBaseURL(url string) *Client {
	c.baseURL = strings.TrimRight(url, "/")
	return c
}

// WithHTTPClient injects a test-controlled HTTP client.
func (c *Client) WithHTTPClient(h *http.Client) *Client {
	c.httpClient = h
	return c
}

// ---------------------------------------------------------------------------
// Pull
// ---------------------------------------------------------------------------

// Spec is the parsed form of a `pull` argument.
//
// Two surface forms:
//   - Repo path with explicit file: "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
//     means look in `bartowski/Llama-3.2-3B-Instruct-GGUF` for the file
//     whose name contains "Q4_K_M" (and `.gguf`).
//   - Alias: "llama3.2:3b" → resolved by the catalog to a (repo, file)
//     tuple.
//
// LocalName is the filename (without `.gguf`) the file gets on disk.
// Defaults to the alias for catalog-resolved specs, else the repo name
// + quant marker.
type Spec struct {
	Repo      string
	FileMatch string // substring to match in the repo's file listing (e.g., "Q4_K_M")
	LocalName string
}

// ParseSpec parses a CLI argument into a Spec.
//
// Forms accepted:
//
//	llama3.2:3b                                            (catalog alias)
//	bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M            (explicit repo:quant)
//	bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
//	                                                       (explicit repo + full filename)
//
// Returns an error if the form is unrecognised.
func ParseSpec(arg string) (Spec, error) {
	if arg == "" {
		return Spec{}, errors.New("empty model spec")
	}

	// Catalog alias? Lookup is opportunistic — anything not in the
	// catalog falls through to the explicit-repo branches.
	if entry, ok := lookupAlias(arg); ok {
		return Spec{
			Repo:      entry.Repo,
			FileMatch: entry.FileMatch,
			LocalName: entry.LocalName,
		}, nil
	}

	// repo/path:quant form (most common explicit shape)
	if idx := strings.LastIndex(arg, ":"); idx > 0 {
		repo := arg[:idx]
		match := arg[idx+1:]
		if !strings.Contains(repo, "/") {
			return Spec{}, fmt.Errorf("invalid spec %q: repo must be <user>/<name>", arg)
		}
		return Spec{
			Repo:      repo,
			FileMatch: match,
			LocalName: localNameFromRepoMatch(repo, match),
		}, nil
	}

	// repo/path/full-filename.gguf form
	if strings.HasSuffix(arg, ".gguf") && strings.Count(arg, "/") >= 2 {
		// last segment is the filename, rest is the repo
		lastSlash := strings.LastIndex(arg, "/")
		repo := arg[:lastSlash]
		filename := arg[lastSlash+1:]
		return Spec{
			Repo:      repo,
			FileMatch: strings.TrimSuffix(filename, ".gguf"),
			LocalName: strings.TrimSuffix(filename, ".gguf"),
		}, nil
	}

	return Spec{}, fmt.Errorf("unrecognised model spec %q. Expected `<alias>` (e.g. `llama3.2:3b`), `<user>/<repo>:<quant>` (e.g. `bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M`), or `<user>/<repo>/<file>.gguf`", arg)
}

// localNameFromRepoMatch picks a reasonable on-disk basename from a
// (repo, quant-match) tuple when the operator didn't override.
// "bartowski/Llama-3.2-3B-Instruct-GGUF" + "Q4_K_M" -> "llama-3.2-3b-instruct-q4_k_m".
func localNameFromRepoMatch(repo, match string) string {
	parts := strings.Split(repo, "/")
	last := parts[len(parts)-1]
	// Strip a trailing "-GGUF" / "-gguf" suffix (cosmetic — most HF
	// repos hosting GGUF quants have it).
	last = strings.TrimSuffix(last, "-GGUF")
	last = strings.TrimSuffix(last, "-gguf")
	if match == "" {
		return strings.ToLower(last)
	}
	return strings.ToLower(last + "-" + match)
}

// hfTreeEntry is one file in an HF repo tree.
type hfTreeEntry struct {
	Path string `json:"path"`
	Type string `json:"type"`
	Size int64  `json:"size"`
	LFS  *struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"lfs,omitempty"`
}

// resolveSpec turns a Spec into the concrete HF tree entry to download.
// Walks the repo's file list, picks the first .gguf file whose name
// contains the FileMatch substring.
func (c *Client) resolveSpec(ctx context.Context, spec Spec) (hfTreeEntry, error) {
	url := fmt.Sprintf("%s/api/models/%s/tree/main", c.baseURL, spec.Repo)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if c.hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.hfToken)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return hfTreeEntry{}, fmt.Errorf("list HF repo %s: %w", spec.Repo, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return hfTreeEntry{}, fmt.Errorf("HF repo %q not found (check the spelling — HF is case-sensitive)", spec.Repo)
	}
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return hfTreeEntry{}, fmt.Errorf("HF repo %q requires auth — set HF_TOKEN (status %d)", spec.Repo, resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return hfTreeEntry{}, fmt.Errorf("HF list %s: status %d: %s", spec.Repo, resp.StatusCode, string(body))
	}

	var entries []hfTreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return hfTreeEntry{}, fmt.Errorf("decode HF tree: %w", err)
	}

	// Prefer the smallest .gguf file matching the substring. The
	// "smallest" tie-break helps when a repo hosts both Q4_K_M and
	// Q4_K_M_L (the L-suffixed one is larger) and the operator typed
	// just "Q4_K_M".
	var best *hfTreeEntry
	for i, e := range entries {
		if e.Type != "file" {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		if spec.FileMatch != "" && !strings.Contains(strings.ToLower(e.Path), strings.ToLower(spec.FileMatch)) {
			continue
		}
		if best == nil || e.Size < best.Size {
			best = &entries[i]
		}
	}
	if best == nil {
		// Build a helpful error listing available GGUFs.
		var available []string
		for _, e := range entries {
			if e.Type == "file" && strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
				available = append(available, e.Path)
			}
		}
		if len(available) == 0 {
			return hfTreeEntry{}, fmt.Errorf("no .gguf files in HF repo %q", spec.Repo)
		}
		return hfTreeEntry{}, fmt.Errorf(
			"no GGUF matching %q in %s. Available:\n  %s",
			spec.FileMatch, spec.Repo, strings.Join(available, "\n  "))
	}
	return *best, nil
}

// Pull downloads the GGUF for spec into c.modelsDir/{LocalName}.gguf,
// atomically. progress is optional; pass nil to skip per-chunk reporting.
//
// Returns the final on-disk path. Idempotent: if the file already
// exists and has the matching SHA256, returns immediately.
func (c *Client) Pull(ctx context.Context, spec Spec, progress ProgressFn) (string, error) {
	if err := os.MkdirAll(c.modelsDir, 0o755); err != nil {
		return "", fmt.Errorf("create models dir: %w", err)
	}

	entry, err := c.resolveSpec(ctx, spec)
	if err != nil {
		return "", err
	}

	localName := spec.LocalName
	if localName == "" {
		localName = strings.TrimSuffix(filepath.Base(entry.Path), ".gguf")
	}
	finalPath := filepath.Join(c.modelsDir, localName+".gguf")

	// Idempotency: if final file exists AND size matches AND (if HF
	// gave us a SHA) sha matches, skip the download.
	if info, err := os.Stat(finalPath); err == nil {
		expectedSize := entry.Size
		if entry.LFS != nil {
			expectedSize = entry.LFS.Size
		}
		if info.Size() == expectedSize {
			if entry.LFS == nil || entry.LFS.SHA256 == "" {
				return finalPath, nil // size matches; no SHA to verify
			}
			actualSha, shaErr := fileSHA256(finalPath)
			if shaErr == nil && actualSha == entry.LFS.SHA256 {
				return finalPath, nil
			}
		}
		// Size or SHA mismatch — fall through to redownload.
	}

	// Download to a `.qf-partial` tmpfile; rename on success.
	tmpPath := finalPath + PartialSuffix
	downloadURL := fmt.Sprintf("%s/%s/resolve/main/%s", c.baseURL, spec.Repo, entry.Path)

	if err := c.download(ctx, downloadURL, tmpPath, entry, progress); err != nil {
		// Don't auto-remove the partial — the next Pull resumes from
		// where this one left off (if HTTP Range support is intact).
		return "", err
	}

	// Verify SHA256 if HF gave us one.
	if entry.LFS != nil && entry.LFS.SHA256 != "" {
		actualSha, shaErr := fileSHA256(tmpPath)
		if shaErr != nil {
			return "", fmt.Errorf("compute SHA256 of downloaded file: %w", shaErr)
		}
		if actualSha != entry.LFS.SHA256 {
			_ = os.Remove(tmpPath)
			return "", fmt.Errorf(
				"SHA256 mismatch — refusing to install. Expected %s, got %s. "+
					"This is either a corrupt download or (unlikely) HF served wrong bytes. Re-run to retry.",
				entry.LFS.SHA256, actualSha)
		}
	}

	// Atomic rename.
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return "", fmt.Errorf("finalise download: %w", err)
	}
	return finalPath, nil
}

// download streams the URL to tmpPath, resuming if tmpPath exists.
func (c *Client) download(ctx context.Context, url, tmpPath string, entry hfTreeEntry, progress ProgressFn) error {
	expectedSize := entry.Size
	if entry.LFS != nil {
		expectedSize = entry.LFS.Size
	}

	// Resume? If a partial exists with size <= expected, send a Range
	// request and append to it.
	var startOffset int64
	if info, err := os.Stat(tmpPath); err == nil {
		if info.Size() > 0 && info.Size() < expectedSize {
			startOffset = info.Size()
		} else if info.Size() >= expectedSize {
			// Stale full-size partial — wipe and restart.
			_ = os.Remove(tmpPath)
		}
	}

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if c.hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.hfToken)
	}
	if startOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startOffset))
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored Range or fresh download — open fresh tmpfile.
		startOffset = 0
	case http.StatusPartialContent:
		// Resume confirmed.
	case http.StatusNotFound:
		return fmt.Errorf("HF returned 404 for %s — repo or file moved?", url)
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("download %s: status %d: %s", url, resp.StatusCode, string(body))
	}

	flag := os.O_CREATE | os.O_WRONLY
	if startOffset > 0 {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	f, err := os.OpenFile(tmpPath, flag, 0o644)
	if err != nil {
		return fmt.Errorf("open tmpfile: %w", err)
	}
	defer f.Close()

	// Copy with progress reporting at most every ~1 MB so the callback
	// doesn't drown a TTY.
	written := startOffset
	buf := make([]byte, 256*1024) // 256 KB read chunk
	var lastReport int64 = startOffset
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("write tmpfile: %w", writeErr)
			}
			written += int64(n)
			if progress != nil && written-lastReport >= 1024*1024 {
				progress(written, expectedSize)
				lastReport = written
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("read download body: %w", readErr)
		}
	}
	if progress != nil {
		progress(written, expectedSize) // final tick
	}

	if err := f.Sync(); err != nil {
		return fmt.Errorf("fsync tmpfile: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// List + Remove
// ---------------------------------------------------------------------------

// List enumerates GGUF files in modelsDir. Sorted by name.
func List(modelsDir string) ([]Model, error) {
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // empty registry, not an error
		}
		return nil, fmt.Errorf("read models dir: %w", err)
	}
	var out []Model
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			continue
		}
		full := filepath.Join(modelsDir, name)
		info, err := e.Info()
		if err != nil {
			continue
		}
		m := Model{
			Name:       strings.TrimSuffix(name, ".gguf"),
			Path:       full,
			ModifiedAt: info.ModTime(),
			Symlink:    info.Mode()&os.ModeSymlink != 0,
		}
		// Symlinks: stat the target for real size.
		if m.Symlink {
			if targetInfo, statErr := os.Stat(full); statErr == nil {
				m.SizeBytes = targetInfo.Size()
			}
		} else {
			m.SizeBytes = info.Size()
		}
		out = append(out, m)
	}
	return out, nil
}

// Remove deletes the GGUF named `name` (without .gguf suffix) from
// modelsDir. Returns an error if it doesn't exist. Follows symlinks
// safely — removes the symlink itself, never the target (which might
// be a shared Ollama blob).
func Remove(modelsDir, name string) error {
	if strings.ContainsAny(name, "/\\") {
		return fmt.Errorf("model name %q must not contain path separators", name)
	}
	path := filepath.Join(modelsDir, name+".gguf")
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no model %q in %s", name, modelsDir)
		}
		return err
	}
	// Use os.Remove (does not follow symlinks for the inode itself —
	// removes the symlink entry without touching the target).
	_ = info
	return os.Remove(path)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
