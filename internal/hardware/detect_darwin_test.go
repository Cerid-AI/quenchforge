// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

// Tests for the darwin-only classifier. Lives behind the same build tag as
// detect_darwin.go so non-darwin CI doesn't try to compile them.

package hardware

import "testing"

func TestClassifyProfileBuckets(t *testing.T) {
	cases := []struct {
		gpu      string
		hasMetal bool
		want     Profile
	}{
		{"AMD Radeon Pro Vega II Duo", true, ProfileVegaPro},
		{"AMD Radeon Pro Vega II", true, ProfileVegaPro},
		{"AMD Radeon Pro W6800X Duo", true, ProfileW6800X},
		{"AMD Radeon Pro 5500M", true, ProfileRDNA1},
		{"AMD Radeon RX 5700 XT", true, ProfileRDNA1},
		{"AMD Radeon RX 6700M", true, ProfileRDNA2},
		{"Intel Iris Plus Graphics", true, ProfileIGPU},
		{"Intel UHD Graphics 630", true, ProfileIGPU},
		{"some-unknown-gpu", true, ProfileMetalUnknown}, // conservative: no AMD-discrete tuning on a guess
		{"", false, ProfileCPU},
	}
	for _, tc := range cases {
		got := classifyProfile(Info{GPU: tc.gpu, HasMetal: tc.hasMetal})
		if got != tc.want {
			t.Errorf("classifyProfile(%q, hasMetal=%v) = %q, want %q",
				tc.gpu, tc.hasMetal, got, tc.want)
		}
	}
}

func TestClassifyProfileAppleSilicon(t *testing.T) {
	info := Info{
		GPU:      "Apple M2 Max",
		HasMetal: true,
		Devices:  []Device{{Name: "Apple M2 Max", AppleSilicon: true, VRAMGB: 64}},
	}
	if got := classifyProfile(info); got != ProfileAppleSilicon {
		t.Errorf("AppleSilicon classify = %q, want %q", got, ProfileAppleSilicon)
	}
}

// An Apple Silicon Mac whose vendor key drifts (system_profiler schema change,
// or a model string without "Apple M") falls through every named bucket. It
// used to land on vega-pro, which forces chat to the CPU with --gpu-layers 0
// and disables Metal concurrency — a large unexplained slowdown presented as
// normal operation.
func TestClassifyProfile_UnrecognisedMetalDeviceGetsConservativeDefaults(t *testing.T) {
	info := Info{
		GPU:      "Apple GPU",
		HasMetal: true,
		Devices:  []Device{{Name: "Apple GPU", VRAMGB: 32}},
	}
	if got := classifyProfile(info); got != ProfileMetalUnknown {
		t.Errorf("classifyProfile(unrecognised Metal device) = %q, want %q", got, ProfileMetalUnknown)
	}
}
