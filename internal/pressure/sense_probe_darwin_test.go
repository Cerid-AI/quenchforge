// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package pressure

import (
	"os"
	"path/filepath"
	"testing"
)

// The governor is spawned by launchd, whose PATH the shipped LaunchAgent sets
// to /usr/local/bin:/usr/bin:/bin — ioreg and sysctl live in /usr/sbin. A
// bare command name therefore resolves to nothing on a stock install, the
// display probe fails, and the safety system silently reports "headless".
func TestProbeToolsAreAbsolutePaths(t *testing.T) {
	for _, p := range []string{ioregPath, sysctlPath} {
		if !filepath.IsAbs(p) {
			t.Errorf("probe tool %q is not an absolute path; it resolves through "+
				"the launchd PATH, which does not include /usr/sbin", p)
			continue
		}
		if _, err := os.Stat(p); err != nil {
			t.Errorf("probe tool %q does not exist: %v", p, err)
		}
	}
}

func TestSensorReportsProbeFailure(t *testing.T) {
	prev := ioregPath
	ioregPath = filepath.Join(t.TempDir(), "no-such-ioreg")
	defer func() { ioregPath = prev }()

	r := osSensor{}.Read()
	if !r.ProbeFailed {
		t.Fatalf("ioreg is missing but the reading claims a good probe: %+v", r)
	}
}
