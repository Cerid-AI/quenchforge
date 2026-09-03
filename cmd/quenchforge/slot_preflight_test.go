// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
)

func TestPreflightSlotModels_MissingRerankIsAnError(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{
		ModelsDir:    tmp,
		DefaultModel: "chat-model",
		RerankModel:  "bge-reranker-v2-m3",
		RerankPort:   11502,
	}
	makeFakeGGUF(t, tmp, "chat-model", 1<<20)

	var buf bytes.Buffer
	unavailable := preflightSlotModels(cfg, cfg.DefaultModel, true, &buf)

	if unavailable[gateway.KindRerank] == nil {
		t.Fatalf("rerank model is absent from %s but the pre-flight reported it available", tmp)
	}
	if unavailable[gateway.KindChat] != nil {
		t.Errorf("chat model is present; want available, got %v", unavailable[gateway.KindChat])
	}
	out := buf.String()
	if !strings.Contains(out, "ERROR") {
		t.Errorf("missing model must be reported as an error, not a warning; got:\n%s", out)
	}
	for _, want := range []string{"rerank", "bge-reranker-v2-m3", "QUENCHFORGE_RERANK_MODEL", "/v1/rerank", "quenchforge pull"} {
		if !strings.Contains(out, want) {
			t.Errorf("report is missing %q; got:\n%s", want, out)
		}
	}
}

func TestPreflightSlotModels_AllPresentIsSilent(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{
		ModelsDir:      tmp,
		DefaultModel:   "chat-model",
		EmbedModel:     "embed-model",
		CodeEmbedModel: "code-model",
		RerankModel:    "rerank-model",
	}
	for _, n := range []string{"chat-model", "embed-model", "code-model", "rerank-model"} {
		makeFakeGGUF(t, tmp, n, 1<<20)
	}

	var buf bytes.Buffer
	unavailable := preflightSlotModels(cfg, cfg.DefaultModel, true, &buf)
	if len(unavailable) != 0 {
		t.Fatalf("every configured model exists; want no unavailable slots, got %v", unavailable)
	}
	if buf.String() != "" {
		t.Errorf("no output expected when every slot resolves; got:\n%s", buf.String())
	}
}

func TestPreflightSlotModels_ChatSuppressedWhenNoSlot(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{ModelsDir: tmp, DefaultModel: "absent-chat-model"}

	var buf bytes.Buffer
	unavailable := preflightSlotModels(cfg, cfg.DefaultModel, false, &buf)
	if len(unavailable) != 0 {
		t.Fatalf("--no-slot suppresses the chat slot; want no findings, got %v", unavailable)
	}
}
