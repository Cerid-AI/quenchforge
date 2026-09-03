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

func TestUnknownGPUNotice(t *testing.T) {
	got := unknownGPUNotice(hardware.Info{
		Profile: hardware.ProfileMetalUnknown, GPU: "AMD Radeon Pro W7900X", GPUVRAMGB: 48,
	})
	for _, want := range []string{"AMD Radeon Pro W7900X", "upstream defaults", "hardware_profile"} {
		if !strings.Contains(got, want) {
			t.Errorf("notice missing %q: %q", want, got)
		}
	}
	if n := unknownGPUNotice(hardware.Info{Profile: hardware.ProfileVegaPro, GPU: "Vega II"}); n != "" {
		t.Errorf("recognised profile must not warn: %q", n)
	}
}
