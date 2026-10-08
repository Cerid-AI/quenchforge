// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"net"
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

func TestPreflightSlotModels_MissingBackgroundIsAnError(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{
		ModelsDir:       tmp,
		DefaultModel:    "chat-model",
		BackgroundModel: "qwen2.5-3b-instruct-q4_k_m",
	}
	makeFakeGGUF(t, tmp, "chat-model", 1<<20)

	var buf bytes.Buffer
	unavailable := preflightSlotModels(cfg, cfg.DefaultModel, true, &buf)

	if unavailable[gateway.KindBackground] == nil {
		t.Fatalf("background model is absent from %s but the pre-flight reported it available", tmp)
	}
	for _, want := range []string{"background", "QUENCHFORGE_BACKGROUND_MODEL", "qwen2.5-3b-instruct-q4_k_m"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("report is missing %q; got:\n%s", want, buf.String())
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

// ---------------------------------------------------------------------------
// doctor — configured-slot vs model-registry cross-check.
// ---------------------------------------------------------------------------

func TestSlotLine_MarksAbsentModelMissing(t *testing.T) {
	tmp := t.TempDir()
	makeFakeGGUF(t, tmp, "present-model", 1<<20)

	if got := slotLine(tmp, "present-model", 11502); strings.Contains(got, "MISSING") {
		t.Errorf("model exists under %s; got %q", tmp, got)
	}
	got := slotLine(tmp, "absent-model", 11502)
	if !strings.Contains(got, "MISSING") {
		t.Errorf("model is absent from %s but doctor renders it as configured: %q", tmp, got)
	}
	if !strings.Contains(got, "absent-model") || !strings.Contains(got, "11502") {
		t.Errorf("slot line lost the model name or port: %q", got)
	}
	if got := slotLine(tmp, "", 11502); strings.Contains(got, "MISSING") {
		t.Errorf("an unconfigured slot is opt-in, not missing: %q", got)
	}
}

func TestDoctor_MarksConfiguredSlotWithAbsentModelMissing(t *testing.T) {
	skipIfNotDarwin(t)
	tmp := t.TempDir()
	t.Setenv("QUENCHFORGE_MODELS_DIR", tmp)
	t.Setenv("QUENCHFORGE_RERANK_MODEL", "bge-reranker-v2-m3")
	// doctor prefers a live gateway's slot report; point it at an address
	// nothing listens on so the process environment above is what it reads.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("QUENCHFORGE_LISTEN_ADDR", addr)

	var stdout, stderr bytes.Buffer
	if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
		t.Fatalf("cmdDoctor: %v", err)
	}
	out := stdout.String()
	line := ""
	for _, l := range strings.Split(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(l), "rerank:") {
			line = l
		}
	}
	if line == "" {
		t.Fatalf("doctor output has no rerank slot line:\n%s", out)
	}
	if !strings.Contains(line, "MISSING") {
		t.Errorf("rerank model is absent from the registry doctor prints below, "+
			"but the slot line reports it as configured: %q", line)
	}
}
