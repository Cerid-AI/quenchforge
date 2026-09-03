// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/hardware"
	"github.com/cerid-ai/quenchforge/internal/pressure"
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

type fakeSensor struct{ r pressure.Reading }

func (f fakeSensor) Read() pressure.Reading { return f.r }

func governorConfig() config.Config {
	return config.Config{
		GPUConcurrencyMax:           6,
		GPUConcurrencyDisplayActive: 1,
		GPUDutyCycleDisplayActive:   0.5,
		GovernorIntervalMS:          1000,
	}
}

func TestStartGovernor_ReportsAFailedPressureProbe(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var out bytes.Buffer
	sched := startGovernor(ctx, fakeSensor{pressure.Reading{ProbeFailed: true, MemPressure: pressure.MemNormal}},
		governorConfig(), &out)

	if !strings.Contains(out.String(), "probe") {
		t.Errorf("a pressure probe that cannot run disables the panic guard silently; "+
			"the governor must say so. Got:\n%s", out.String())
	}
	if sched.Concurrency() != 1 {
		t.Errorf("failed probe must hold the display-active ceiling, got concurrency %d",
			sched.Concurrency())
	}
}

func TestStartGovernor_QuietWhenProbesWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var out bytes.Buffer
	startGovernor(ctx, fakeSensor{pressure.Reading{DisplayActive: false, MemPressure: pressure.MemNormal}},
		governorConfig(), &out)

	if strings.Contains(out.String(), "probe") {
		t.Errorf("healthy probes must not warn. Got:\n%s", out.String())
	}
}
