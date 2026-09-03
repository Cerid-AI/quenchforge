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

// The GPU governor probes ioreg and sysctl, both of which live in /usr/sbin.
// launchd hands the job the PATH from this plist and nothing else, so an
// omitted /usr/sbin switches the compositor-starvation guard off on every
// fresh install.
func TestInstall_PlistPathIncludesUsrSbin(t *testing.T) {
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
	plist, err := os.ReadFile(filepath.Join(home, "Library", "LaunchAgents", plistFilename))
	if err != nil {
		t.Fatalf("read plist: %v", err)
	}
	line := ""
	for _, l := range strings.Split(string(plist), "\n") {
		if strings.Contains(l, "/usr/local/bin:") {
			line = l
		}
	}
	if line == "" {
		t.Fatalf("plist has no PATH value:\n%s", plist)
	}
	if !strings.Contains(line, "/usr/sbin") {
		t.Errorf("LaunchAgent PATH omits /usr/sbin, where ioreg and sysctl live: %s", line)
	}
}
