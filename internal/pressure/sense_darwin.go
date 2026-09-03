// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package pressure

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// NewSensor returns the macOS pressure sensor. It shells out to ioreg and
// sysctl (no CGo — per the repo's CGo-only-in-detect_darwin.go rule) with
// short timeouts so a hung probe can never stall the governor loop.
func NewSensor() Sensor { return osSensor{} }

// Absolute paths, matching detect_darwin.go. Both tools live in /usr/sbin,
// which the shipped LaunchAgent's PATH does not contain: resolving them by
// bare name turned the governor off on every launchd-started install and the
// only symptom was full-throughput inference reported as normal. Package vars
// so tests can point them at a missing binary.
var (
	ioregPath  = "/usr/sbin/ioreg"
	sysctlPath = "/usr/sbin/sysctl"
)

type osSensor struct{}

func (osSensor) Read() Reading {
	active, activeOK := displayActive()
	mem, memOK := memPressure()
	return Reading{
		DisplayActive: active,
		MemPressure:   mem,
		ProbeFailed:   !activeOK || !memOK,
	}
}

const probeTimeout = 2 * time.Second

func probe(name string, args ...string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return "", false
	}
	return string(out), true
}

// displayActive parses the IODisplayWrangler power-management dict from
// `ioreg` and reports true when the display is at full power
// (CurrentPowerState == MaxPowerState). The second return distinguishes a
// probe that ran from one that could not: a headless host and a host where
// ioreg would not execute both read as "no display", but only the first is a
// safe reason to run flat out.
func displayActive() (active, ok bool) {
	out, ran := probe(ioregPath, "-n", "IODisplayWrangler", "-r", "-d", "1")
	if !ran {
		return false, false
	}
	cur, curOK := extractInt(out, `"CurrentPowerState"=`)
	max, maxOK := extractInt(out, `"MaxPowerState"=`)
	if !curOK || !maxOK || max < 1 {
		// ioreg ran and reported no display wrangler: genuinely headless.
		return false, true
	}
	return cur >= max, true
}

// memPressure reads kern.memorystatus_vm_pressure_level (1/2/4). Returns
// MemNormal plus ok=false when the probe could not run.
func memPressure() (level int, ok bool) {
	out, ran := probe(sysctlPath, "-n", "kern.memorystatus_vm_pressure_level")
	if !ran {
		return MemNormal, false
	}
	v, err := strconv.Atoi(strings.TrimSpace(out))
	if err != nil || v < 1 {
		return MemNormal, true
	}
	return v, true
}

// extractInt finds key in s and parses the run of digits immediately
// following it. Used to pluck values out of ioreg's flat dict rendering
// without a full plist parse.
func extractInt(s, key string) (int, bool) {
	i := strings.Index(s, key)
	if i < 0 {
		return 0, false
	}
	j := i + len(key)
	k := j
	for k < len(s) && s[k] >= '0' && s[k] <= '9' {
		k++
	}
	if k == j {
		return 0, false
	}
	v, err := strconv.Atoi(s[j:k])
	if err != nil {
		return 0, false
	}
	return v, true
}
