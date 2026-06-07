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

type osSensor struct{}

func (osSensor) Read() Reading {
	return Reading{
		DisplayActive: displayActive(),
		MemPressure:   memPressure(),
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
// (CurrentPowerState == MaxPowerState). When the node is absent (headless),
// the probe fails, or the display is asleep (CurrentPowerState < Max), it
// returns false — those hosts get full inference throughput.
func displayActive() bool {
	out, ok := probe("ioreg", "-n", "IODisplayWrangler", "-r", "-d", "1")
	if !ok {
		return false
	}
	cur, curOK := extractInt(out, `"CurrentPowerState"=`)
	max, maxOK := extractInt(out, `"MaxPowerState"=`)
	if !curOK || !maxOK || max < 1 {
		return false
	}
	return cur >= max
}

// memPressure reads kern.memorystatus_vm_pressure_level (1/2/4). Returns
// MemNormal on any failure so a probe miss never throttles.
func memPressure() int {
	out, ok := probe("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
	if !ok {
		return MemNormal
	}
	v, err := strconv.Atoi(strings.TrimSpace(out))
	if err != nil || v < 1 {
		return MemNormal
	}
	return v
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
