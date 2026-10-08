// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Startup pre-flight for configured slot models.
//
// A slot whose model env var is set is a slot the operator asked for. When
// the GGUF behind it is absent the slot cannot serve, and every request on
// that lane 503s for the lifetime of the process. The only signal used to be
// one `warning:` line per boot on stderr, which is how the rerank lane stayed
// dead for days while `/` reported the slot as simply "not configured".
//
// This pre-flight runs before any child is spawned. It resolves each
// configured model, reports an absent one as an ERROR naming the lane, the
// env var and the fix, and returns the affected kinds so cmdServe skips the
// spawn entirely — no child, no upstream registration, no ambiguity about
// whether the lane is live.

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
)

// slotModel is one configured GGUF-backed slot: the kind the gateway routes
// to, the env var that configured it, and the lane a consumer calls.
type slotModel struct {
	Kind   gateway.SlotKind
	Name   string
	EnvVar string
	Lane   string
	Model  string
}

// configuredSlotModels lists the GGUF-backed slots this config asks for.
// chatModel is the effective chat model (the --model flag wins over
// cfg.DefaultModel); chatEnabled is false under --no-slot.
//
// whisper / image-gen / TTS are deliberately absent: they load through
// different binaries whose model argument is a path rather than a registry
// name, so their absence is caught by the per-binary launch path.
func configuredSlotModels(cfg config.Config, chatModel string, chatEnabled bool) []slotModel {
	all := []slotModel{
		{gateway.KindChat, "chat", "QUENCHFORGE_DEFAULT_MODEL", "/api/chat + /v1/chat/completions", chatModel},
		{gateway.KindEmbed, "embed", "QUENCHFORGE_EMBED_MODEL", "/api/embeddings + /v1/embeddings", cfg.EmbedModel},
		{gateway.KindCodeEmbed, "code-embed", "QUENCHFORGE_CODE_EMBED_MODEL", "/v1/embeddings (model=" + cfg.CodeEmbedModel + ")", cfg.CodeEmbedModel},
		{gateway.KindBackground, "background", "QUENCHFORGE_BACKGROUND_MODEL", "/api/chat + /v1/chat/completions (model=" + cfg.BackgroundModel + ")", cfg.BackgroundModel},
		{gateway.KindRerank, "rerank", "QUENCHFORGE_RERANK_MODEL", "/v1/rerank", cfg.RerankModel},
	}
	out := make([]slotModel, 0, len(all))
	for _, s := range all {
		if s.Model == "" {
			continue
		}
		if s.Kind == gateway.KindChat && !chatEnabled {
			continue
		}
		out = append(out, s)
	}
	return out
}

// missingSlotModels returns the resolution error for every configured slot
// whose model cannot be found under cfg.ModelsDir.
func missingSlotModels(cfg config.Config, slots []slotModel) map[gateway.SlotKind]error {
	missing := map[gateway.SlotKind]error{}
	for _, s := range slots {
		if _, err := resolveSlotModel(cfg.ModelsDir, s.Model); err != nil {
			missing[s.Kind] = err
		}
	}
	return missing
}

// resolveSlotModel resolves a slot's model argument to a path on disk. Names
// resolve against modelsDir; an absolute path (whisper / sd / bark style) is
// checked as given.
func resolveSlotModel(modelsDir, model string) (string, error) {
	if filepath.IsAbs(model) {
		if _, err := os.Stat(model); err != nil {
			return "", fmt.Errorf("model %q not found", model)
		}
		return model, nil
	}
	return resolveModelPath(modelsDir, model)
}

// preflightSlotModels reports every configured slot whose model is absent and
// returns those kinds keyed to the resolution error. Callers must not start a
// slot that appears in the result: it cannot load, and starting it anyway
// leaves an unserviceable lane looking like a transient failure.
func preflightSlotModels(cfg config.Config, chatModel string, chatEnabled bool, w io.Writer) map[gateway.SlotKind]error {
	slots := configuredSlotModels(cfg, chatModel, chatEnabled)
	missing := missingSlotModels(cfg, slots)
	if len(missing) > 0 {
		fmt.Fprint(w, formatMissingSlotModels(cfg.ModelsDir, slots, missing))
	}
	return missing
}

// formatMissingSlotModels renders the operator-facing report. It names the
// lane a consumer will see fail, the env var that configured the slot, and
// both fixes (install the model, or stop configuring the slot).
func formatMissingSlotModels(modelsDir string, slots []slotModel, missing map[gateway.SlotKind]error) string {
	var affected []slotModel
	for _, s := range slots {
		if missing[s.Kind] != nil {
			affected = append(affected, s)
		}
	}
	sort.Slice(affected, func(i, j int) bool { return affected[i].Name < affected[j].Name })

	var sb strings.Builder
	noun := "slot is"
	if len(affected) > 1 {
		noun = "slots are"
	}
	fmt.Fprintf(&sb, "quenchforge: ERROR: %d configured %s missing its model on disk.\n",
		len(affected), noun)
	fmt.Fprintln(&sb, "  The slot will NOT be started and its lane returns HTTP 503 until this is fixed.")
	fmt.Fprintln(&sb)
	for _, s := range affected {
		fmt.Fprintf(&sb, "    %-11s model=%q (%s)\n", s.Name+":", s.Model, s.EnvVar)
		fmt.Fprintf(&sb, "                lane: %s\n", s.Lane)
		fmt.Fprintf(&sb, "                fix:  quenchforge pull %s   # or place the GGUF under the models dir\n", s.Model)
		fmt.Fprintf(&sb, "                      unset %s   # to stop configuring this slot\n", s.EnvVar)
	}
	fmt.Fprintln(&sb)
	fmt.Fprintf(&sb, "  models dir: %s\n", modelsDir)
	fmt.Fprintln(&sb, "  `quenchforge list` shows what is installed; `quenchforge doctor` marks the slot MISSING.")
	return sb.String()
}
