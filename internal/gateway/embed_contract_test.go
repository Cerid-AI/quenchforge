// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"net/http"
	"strings"
	"testing"
)

// A code-embed request answered by the general-text embedder returns
// vectors from a different model in a different space — silently unusable,
// and poison if the caller writes them to an index.
func TestCodeEmbedWithoutSlotIsRefusedNotSubstituted(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.EmbedModel = "nomic-embed-text-v1.5"
	cfg.CodeEmbedModel = "nomic-embed-code-v1"
	g := newRunningGateway(t, cfg)
	embedResp := `{"object":"list","model":"nomic-embed-text-v1.5","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`
	upstream, _ := newCapturingUpstream(t, http.StatusOK, embedResp, "application/json")
	if err := g.SetUpstream(KindEmbed, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	for _, path := range []string{"/api/embeddings", "/v1/embeddings"} {
		status, body := postJSON(t, "http://"+cfg.ListenAddr+path,
			`{"model":"nomic-embed-code-v1","input":"func main() {}"}`)
		if status != http.StatusServiceUnavailable {
			t.Errorf("%s: status = %d body=%s, want 503 rather than a general-text substitution",
				path, status, body)
		}
		if !strings.Contains(body, string(KindCodeEmbed)) {
			t.Errorf("%s: body = %s, want it to name the missing code-embed slot", path, body)
		}
	}
}

// A general-text embed request is unaffected by the code-embed guard.
func TestGeneralEmbedStillServedWhenCodeEmbedConfigured(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.EmbedModel = "nomic-embed-text-v1.5"
	cfg.CodeEmbedModel = "nomic-embed-code-v1"
	g := newRunningGateway(t, cfg)
	embedResp := `{"object":"list","model":"nomic-embed-text-v1.5","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}`
	upstream, _ := newCapturingUpstream(t, http.StatusOK, embedResp, "application/json")
	if err := g.SetUpstream(KindEmbed, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	status, body := postJSON(t, "http://"+cfg.ListenAddr+"/api/embeddings",
		`{"model":"nomic-embed-text-v1.5","prompt":"hello"}`)
	if status != http.StatusOK {
		t.Errorf("status = %d body=%s, want 200", status, body)
	}
}
