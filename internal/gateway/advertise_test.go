// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func getJSON(t *testing.T, url string) map[string]any {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode %s: %v body=%s", url, err, raw)
	}
	return out
}

// GET / is the machine-readable capability advertisement. It claimed
// imagegen and TTS were 501 reservations while both are live proxy routes,
// and omitted the native paths entirely.
func TestRootAdvertisesTheRoutesThatAreActuallyMounted(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	newRunningGateway(t, cfg)

	root := getJSON(t, "http://"+cfg.ListenAddr+"/")
	rawRoutes, ok := root["routes"].([]any)
	if !ok {
		t.Fatalf("/ has no routes list: %v", root)
	}
	routes := make([]string, 0, len(rawRoutes))
	for _, r := range rawRoutes {
		routes = append(routes, r.(string))
	}
	joined := strings.Join(routes, "\n")
	if strings.Contains(joined, "501 reserved") {
		t.Errorf("/ still advertises live routes as 501 reservations:\n%s", joined)
	}
	for _, want := range []string{
		"GET /",
		"GET /health",
		"GET /api/tags",
		"POST /api/chat",
		"POST /api/generate",
		"POST /v1/chat/completions",
		"POST /api/embeddings",
		"POST /api/embed",
		"POST /v1/embeddings",
		"POST /v1/rerank",
		"POST /rerank",
		"POST /v1/audio/transcriptions",
		"POST /v1/audio/translations",
		"POST /inference",
		"POST /v1/images/generations",
		"POST /sdapi/v1/txt2img",
		"POST /sdapi/v1/img2img",
		"POST /v1/audio/speech",
		"POST /tts",
	} {
		found := false
		for _, r := range routes {
			if r == want || strings.HasPrefix(r, want+" ") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("/ does not advertise mounted route %q:\n%s", want, joined)
		}
	}
	// Every advertised route must actually be mounted (no 404).
	for _, r := range routes {
		fields := strings.Fields(r)
		if len(fields) < 2 || fields[0] != "GET" {
			continue
		}
		resp, err := http.Get("http://" + cfg.ListenAddr + fields[1])
		if err != nil {
			t.Fatalf("GET %s: %v", fields[1], err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode == http.StatusNotFound {
			t.Errorf("advertised route %q is not mounted (404)", r)
		}
	}
}
