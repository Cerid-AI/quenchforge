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
