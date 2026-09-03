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
		Max:               cfg.GPUConcurrencyMax,
		DisplayActive:     cfg.GPUConcurrencyDisplayActive,
		DisplayActiveDuty: cfg.GPUDutyCycleDisplayActive,
	}
	// Start at the conservative display-active plan so we never open the
	// floodgates before the first reading lands.
	startPlan := limits.For(pressure.Reading{DisplayActive: true, MemPressure: pressure.MemNormal})
	sched := scheduler.New(startPlan.Concurrency)
	sched.SetDutyCycle(startPlan.Duty)

	interval := time.Duration(cfg.GovernorIntervalMS) * time.Millisecond
	if interval < 250*time.Millisecond {
		interval = 250 * time.Millisecond
	}

	// A probe that cannot run (missing ioreg / sysctl on the launchd PATH)
	// leaves the governor holding the conservative plan on unmeasured data.
	// Say so on every transition: the failure mode it replaces was a safety
	// system reporting normal operation while switched off.
	probeFailed := false
	apply := func() {
		reading := sensor.Read()
		if reading.ProbeFailed != probeFailed {
			probeFailed = reading.ProbeFailed
			if probeFailed {
				fmt.Fprintln(out, "quenchforge: WARNING: GPU governor pressure probe failed "+
					"(ioreg/sysctl did not run) — holding the display-active plan on unmeasured "+
					"data. Check that /usr/sbin is on the daemon's PATH.")
			} else {
				fmt.Fprintln(out, "quenchforge: GPU governor pressure probe recovered")
			}
		}
		plan := limits.For(reading)
		if sched.Concurrency() != plan.Concurrency {
			sched.SetConcurrency(plan.Concurrency)
		}
		if sched.DutyCycle() != plan.Duty {
			sched.SetDutyCycle(plan.Duty)
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

	fmt.Fprintf(out, "quenchforge: GPU governor active (max=%d display-active conc=%d duty=%.2f cooldown<=%dms interval=%s)\n",
		cfg.GPUConcurrencyMax, cfg.GPUConcurrencyDisplayActive, cfg.GPUDutyCycleDisplayActive,
		cfg.GovernorMaxCooldownMS, interval)
	return sched
}
