// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package supervisor

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// RotatingWriter is an io.Writer that rolls the underlying file over
// at a configurable size threshold, keeping a bounded number of
// numbered backup files (file.log, file.log.1, ..., file.log.N).
//
// Used by the supervisor to bound per-slot log growth. llama-server
// chat/embed/rerank slots emit one model-load banner per spawn (multi-KB
// each), and the embed slot under sustained load produces verbose
// per-request output that accumulates to GB-scale in days. Without
// rotation, an unattended quenchforge install consumes the user's
// disk and contributes to the disk-full state that prevents APFS
// from writing kernel panic reports.
//
// The rotator is intentionally minimal: synchronous rotation on the
// write path (not a background goroutine), no compression of backups,
// no time-based policy. Operators who need more (e.g., daily rotation,
// gzip) can layer macOS `newsyslog` on top — the rotator does not own
// the file exclusively in any way that breaks external rotation.
type RotatingWriter struct {
	path     string
	maxBytes int64 // 0 disables rotation
	backups  int

	mu      sync.Mutex
	f       *os.File
	curSize int64
}

// NewRotatingWriter opens path for append. If the file exists, its
// current size becomes the starting curSize. maxBytes <= 0 disables
// rotation (writes append forever).
func NewRotatingWriter(path string, maxBytes int64, backups int) (*RotatingWriter, error) {
	if backups < 0 {
		backups = 0
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("logrotate: mkdir %s: %w", filepath.Dir(path), err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("logrotate: open %s: %w", path, err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("logrotate: stat %s: %w", path, err)
	}
	return &RotatingWriter{
		path:     path,
		maxBytes: maxBytes,
		backups:  backups,
		f:        f,
		curSize:  info.Size(),
	}, nil
}

// Write implements io.Writer. May rotate the file synchronously when
// a write would cause curSize to exceed maxBytes.
func (w *RotatingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.maxBytes > 0 && w.curSize+int64(len(p)) > w.maxBytes {
		if err := w.rotateLocked(); err != nil {
			return 0, err
		}
	}
	n, err := w.f.Write(p)
	w.curSize += int64(n)
	return n, err
}

// Close releases the underlying file. Safe to call multiple times.
func (w *RotatingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := w.f.Close()
	w.f = nil
	return err
}

// rotateLocked: caller holds w.mu. Closes the current file, rolls
// existing backups one slot higher, then reopens path fresh.
//
// Backup naming: file.log → file.log.1 → file.log.2 → ... → file.log.N.
// The oldest is dropped on overflow. If backups == 0, the primary is
// truncated with no backup preserved.
func (w *RotatingWriter) rotateLocked() error {
	if w.f != nil {
		if err := w.f.Close(); err != nil {
			return fmt.Errorf("logrotate: close before rotate: %w", err)
		}
		w.f = nil
	}

	// Walk from oldest to newest, shifting each up one slot.
	// e.g., backups=3: drop .3, rename .2→.3, .1→.2, primary→.1.
	if w.backups > 0 {
		oldest := fmt.Sprintf("%s.%d", w.path, w.backups)
		_ = os.Remove(oldest) // best-effort; missing is fine

		for i := w.backups - 1; i >= 1; i-- {
			src := fmt.Sprintf("%s.%d", w.path, i)
			dst := fmt.Sprintf("%s.%d", w.path, i+1)
			if _, err := os.Stat(src); err == nil {
				if err := os.Rename(src, dst); err != nil {
					return fmt.Errorf("logrotate: rename %s -> %s: %w", src, dst, err)
				}
			}
		}
		// primary → .1
		if _, err := os.Stat(w.path); err == nil {
			if err := os.Rename(w.path, w.path+".1"); err != nil {
				return fmt.Errorf("logrotate: rename %s -> %s.1: %w", w.path, w.path, err)
			}
		}
	} else {
		// backups == 0: just remove primary (it's about to be reopened fresh).
		_ = os.Remove(w.path)
	}

	f, err := os.OpenFile(w.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("logrotate: reopen after rotate: %w", err)
	}
	w.f = f
	w.curSize = 0
	return nil
}
