// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/hardware"
)

func TestGovernorDefaultPerProfile(t *testing.T) {
	cases := map[hardware.Profile]bool{
		hardware.ProfileAppleSilicon: false,
		hardware.ProfileVegaPro:      true,
		hardware.ProfileW6800X:       true,
		hardware.ProfileRDNA1:        true,
		hardware.ProfileRDNA2:        true,
		hardware.ProfileIGPU:         true,
		hardware.ProfileCPU:          true,
		hardware.ProfileUnknown:      true,
	}
	for profile, want := range cases {
		if got := governorEnabled(config.Config{}, profile); got != want {
			t.Errorf("governorEnabled(unset, %s) = %v, want %v", profile, got, want)
		}
	}
}

func TestGovernorExplicitSettingWins(t *testing.T) {
	on, off := true, false
	for _, profile := range []hardware.Profile{hardware.ProfileAppleSilicon, hardware.ProfileVegaPro} {
		if !governorEnabled(config.Config{GovernorEnabled: &on}, profile) {
			t.Errorf("QUENCHFORGE_GOVERNOR=true ignored on %s", profile)
		}
		if governorEnabled(config.Config{GovernorEnabled: &off}, profile) {
			t.Errorf("QUENCHFORGE_GOVERNOR=false ignored on %s", profile)
		}
	}
}
