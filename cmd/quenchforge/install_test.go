// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestInstall_WritesPlistAndPrestartGuard(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("install is macOS-only")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USER", "tester")

	var out, errb bytes.Buffer
	if err := cmdInstall(nil, &out, &errb); err != nil {
		t.Fatalf("cmdInstall: %v (stderr=%s)", err, errb.String())
	}

	// Plist written, REPLACE_ME substituted, ProgramArguments points at the
	// guard under the operator's home (the /Users/$USER convention the
	// template uses for all its paths).
	plist, err := os.ReadFile(filepath.Join(home, "Library", "LaunchAgents", plistFilename))
	if err != nil {
		t.Fatalf("read plist: %v", err)
	}
	ps := string(plist)
	if strings.Contains(ps, "REPLACE_ME") {
		t.Errorf("plist still contains a REPLACE_ME placeholder")
	}
	wantRef := "/Users/tester/" + filepath.ToSlash(prestartGuardRelPath)
	if !strings.Contains(ps, wantRef) {
		t.Errorf("plist ProgramArguments should reference guard %q\n%s", wantRef, ps)
	}

	// Guard written to the operator's HOME, executable, with the eviction
	// logic intact.
	guardAbs := filepath.Join(home, prestartGuardRelPath)
	info, err := os.Stat(guardAbs)
	if err != nil {
		t.Fatalf("stat guard: %v", err)
	}
	if info.Mode().Perm()&0o100 == 0 {
		t.Errorf("guard is not executable: mode %v", info.Mode())
	}
	guard, err := os.ReadFile(guardAbs)
	if err != nil {
		t.Fatalf("read guard: %v", err)
	}
	for _, want := range []string{"com.ollama.ollama", "lsof", "exec "} {
		if !strings.Contains(string(guard), want) {
			t.Errorf("guard script missing expected content %q", want)
		}
	}
}
