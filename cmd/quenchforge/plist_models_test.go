// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/registry"
)

// plistEnv pulls a value out of the embedded LaunchAgent template's
// EnvironmentVariables dict.
func plistEnv(t *testing.T, key string) string {
	t.Helper()
	re := regexp.MustCompile(`(?s)<key>` + regexp.QuoteMeta(key) + `</key>\s*<string>(.*?)</string>`)
	m := re.FindSubmatch(plistTemplate)
	if m == nil {
		t.Fatalf("plist template has no %s entry", key)
	}
	return strings.TrimSpace(string(m[1]))
}

// The installer runs `quenchforge install --force` unattended and downloads no
// models. Every model the LaunchAgent names must therefore be one the
// installer's own next-steps tell the operator to pull, or empty.
func TestPlistTemplate_NamesOnlyModelsTheInstallerProvides(t *testing.T) {
	for _, key := range []string{"QUENCHFORGE_EMBED_MODEL", "QUENCHFORGE_RERANK_MODEL"} {
		if v := plistEnv(t, key); v != "" {
			t.Errorf("%s=%q on a fresh install, but the installer never downloads it: "+
				"the slot starts against a model that is not on disk", key, v)
		}
	}

	installSh, err := os.ReadFile("../../install.sh")
	if err != nil {
		t.Fatalf("read install.sh: %v", err)
	}
	m := regexp.MustCompile(`quenchforge pull ([a-z0-9][a-z0-9.:\-]*)`).FindSubmatch(installSh)
	if m == nil {
		t.Fatalf("install.sh no longer suggests a `quenchforge pull <alias>`")
	}
	alias := string(m[1])

	var want string
	for _, e := range registry.Catalog() {
		if e.Alias == alias {
			want = e.LocalName
		}
	}
	if want == "" {
		t.Fatalf("install.sh suggests `quenchforge pull %s`, which is not a catalog alias", alias)
	}
	if got := plistEnv(t, "QUENCHFORGE_DEFAULT_MODEL"); got != want {
		t.Errorf("LaunchAgent chat model is %q but the installer's own next step pulls %q "+
			"(installs as %q) — first boot cannot find the chat model", got, alias, want)
	}
}
