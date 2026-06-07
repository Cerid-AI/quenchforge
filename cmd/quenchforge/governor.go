// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/pressure"
	"github.com/cerid-ai/quenchforge/internal/scheduler"
)

// startGovernor builds the GPU-admission scheduler and runs the pressure
// governor that resizes it over time. The scheduler is returned so the caller
// can install it on the gateway via SetScheduler.
//
// The governor's job: on a single-GPU Mac, reserve GPU headroom for the
// display compositor whenever a screen is being driven, so sustained
// inference can't starve WindowServer into a kernel-watchdog panic. When the
// host is headless or the display is asleep it restores full throughput, so
// inference-server users see no throttling.
func startGovernor(ctx context.Context, sensor pressure.Sensor, cfg config.Config, out io.Writer) *scheduler.Scheduler {
	limits := pressure.Limits{
		Max:           cfg.GPUConcurrencyMax,
		DisplayActive: cfg.GPUConcurrencyDisplayActive,
	}
	// Start at the conservative display-active ceiling so we never open the
	// floodgates before the first reading lands.
	sched := scheduler.New(limits.Target(pressure.Reading{
		DisplayActive: true,
		MemPressure:   pressure.MemNormal,
	}))

	interval := time.Duration(cfg.GovernorIntervalMS) * time.Millisecond
	if interval < 250*time.Millisecond {
		interval = 250 * time.Millisecond
	}

	apply := func() {
		target := limits.Target(sensor.Read())
		if sched.Concurrency() != target {
			sched.SetConcurrency(target)
		}
	}
	apply() // reflect reality before the first request is served

	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				apply()
			}
		}
	}()

	fmt.Fprintf(out, "quenchforge: GPU governor active (max=%d display-active=%d interval=%s)\n",
		cfg.GPUConcurrencyMax, cfg.GPUConcurrencyDisplayActive, interval)
	return sched
}
