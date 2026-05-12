// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package hardware

import (
	"runtime"
	"strings"
	"testing"
)

func TestDetectReturnsNonZero(t *testing.T) {
	info, err := Detect()
	if err != nil {
		t.Fatalf("Detect: unexpected error: %v", err)
	}
	if info.Profile == "" {
		t.Fatal("Detect returned empty Profile")
	}
	// All platforms should at least know their CPU count via runtime.
	if info.CPUCores == 0 {
		t.Errorf("Detect returned CPUCores=0 (expected >0 from runtime.NumCPU())")
	}
}

func TestProfileStringer(t *testing.T) {
	cases := map[Profile]string{
		ProfileVegaPro:      "vega-pro",
		ProfileW6800X:       "w6800x",
		ProfileRDNA1:        "rdna1",
		ProfileRDNA2:        "rdna2",
		ProfileAppleSilicon: "apple-silicon",
		ProfileIGPU:         "igpu",
		ProfileCPU:          "cpu",
		ProfileUnknown:      "unknown",
	}
	for p, want := range cases {
		if got := p.String(); got != want {
			t.Errorf("Profile %q stringer: got %q, want %q", p, got, want)
		}
	}
}

func TestInfoStringContainsProfile(t *testing.T) {
	info := Info{
		Profile:    ProfileVegaPro,
		OSVersion:  "macOS 14.5",
		CPU:        "Intel Xeon W-3245M",
		CPUCores:   16,
		TotalRAMGB: 96,
		GPU:        "AMD Radeon Pro Vega II",
		GPUVRAMGB:  32,
		HasMetal:   true,
		Devices:    []Device{{Name: "AMD Radeon Pro Vega II", VRAMGB: 32}},
	}
	s := info.String()
	if !strings.Contains(s, "profile=vega-pro") {
		t.Errorf("expected profile token in %q", s)
	}
	if !strings.Contains(s, "vram=32GB") {
		t.Errorf("expected vram token in %q", s)
	}
}

func TestNonDarwinStubProducesCPUProfile(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("only meaningful on non-darwin")
	}
	info, err := Detect()
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if info.Profile != ProfileCPU {
		t.Errorf("non-darwin Detect: profile=%q, want %q", info.Profile, ProfileCPU)
	}
	if info.HasMetal {
		t.Errorf("non-darwin Detect: HasMetal=true, want false")
	}
}

func TestClassifyProfileBuckets(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("classifyProfile lives in detect_darwin.go (build tag)")
	}
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
		{"some-unknown-gpu", true, ProfileVegaPro}, // fallback for discrete-AMD-with-metal
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
	if runtime.GOOS != "darwin" {
		t.Skip("classifyProfile lives in detect_darwin.go (build tag)")
	}
	info := Info{
		GPU:      "Apple M2 Max",
		HasMetal: true,
		Devices:  []Device{{Name: "Apple M2 Max", AppleSilicon: true, VRAMGB: 64}},
	}
	if got := classifyProfile(info); got != ProfileAppleSilicon {
		t.Errorf("AppleSilicon classify = %q, want %q", got, ProfileAppleSilicon)
	}
}
