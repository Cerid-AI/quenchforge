// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"runtime/debug"
	"strings"
	"testing"
)

func buildInfoWith(settings map[string]string) *debug.BuildInfo {
	bi := &debug.BuildInfo{}
	for k, v := range settings {
		bi.Settings = append(bi.Settings, debug.BuildSetting{Key: k, Value: v})
	}
	return bi
}

func TestBuildIdentity_StampedValuesWin(t *testing.T) {
	v, c := buildIdentity("0.10.1", "835dbf9", buildInfoWith(map[string]string{
		"vcs.revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
	}), true)
	if v != "0.10.1" || c != "835dbf9" {
		t.Errorf("buildIdentity = %q/%q, want the ldflag values", v, c)
	}
}

func TestBuildIdentity_UnstampedBuildReportsItsCommit(t *testing.T) {
	v, c := buildIdentity(devVersion, "unknown", buildInfoWith(map[string]string{
		"vcs.revision": "835dbf9aa1b2c3d4e5f60718293a4b5c6d7e8f90",
		"vcs.modified": "false",
	}), true)
	if v == devVersion {
		t.Errorf("an unstamped build must not report a bare %q — the maintainer "+
			"cannot tell which code the operator is running", devVersion)
	}
	if !strings.Contains(v, "835dbf9") {
		t.Errorf("version %q does not carry the revision", v)
	}
	if c != "835dbf9aa1b2c3d4e5f60718293a4b5c6d7e8f90" {
		t.Errorf("commit = %q, want the full revision", c)
	}
	if strings.Contains(v, "dirty") {
		t.Errorf("clean tree reported as dirty: %q", v)
	}
}

func TestBuildIdentity_MarksAModifiedTree(t *testing.T) {
	v, _ := buildIdentity(devVersion, "unknown", buildInfoWith(map[string]string{
		"vcs.revision": "835dbf9aa1b2c3d4e5f60718293a4b5c6d7e8f90",
		"vcs.modified": "true",
	}), true)
	if !strings.Contains(v, "dirty") {
		t.Errorf("version %q does not mark the uncommitted tree", v)
	}
}

func TestBuildIdentity_NoBuildInfo(t *testing.T) {
	v, c := buildIdentity(devVersion, "unknown", nil, false)
	if !strings.Contains(v, "unversioned") {
		t.Errorf("version %q should say the build carries no version at all", v)
	}
	if c != "unknown" {
		t.Errorf("commit = %q, want unknown", c)
	}
}

// The hardcoded default must never be a version number: a stale literal is
// indistinguishable from a real release in a doctor paste.
func TestVersionDefaultIsNotAVersionNumber(t *testing.T) {
	if strings.ContainsAny(devVersion, "0123456789") {
		t.Errorf("default Version %q looks like a release number", devVersion)
	}
}
