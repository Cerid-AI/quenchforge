// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin

// Non-Darwin stub. Quenchforge is macOS-only by design (Metal is the
// moat), so this file's only job is to keep `go test ./...` and
// `go build ./...` green on Linux CI runners — the runners that compile
// the unit tests for the cross-platform parts of the binary even though
// the resulting artifact only ever ships for darwin/{amd64,arm64}.
//
// detect() returns a CPU-only profile so anything that consumes Info
// doesn't have to special-case the non-Darwin path.
package hardware

import (
	"os"
	"runtime"
)

func detect() (Info, error) {
	cores := runtime.NumCPU()
	return Info{
		Profile:    ProfileCPU,
		OSVersion:  runtime.GOOS,
		CPU:        "unknown (non-darwin stub)",
		CPUCores:   cores,
		TotalRAMGB: linuxRAMGB(),
		// Devices intentionally empty — non-Darwin code paths don't read GPUs.
	}, nil
}

// linuxRAMGB peeks at /proc/meminfo when available, returning 0 otherwise.
// Doesn't shell out — keeps the stub dependency-free.
func linuxRAMGB() int {
	if _, err := os.Stat("/proc/meminfo"); err != nil {
		return 0
	}
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	// /proc/meminfo's first line: "MemTotal:       16384324 kB"
	const prefix = "MemTotal:"
	for i := 0; i < len(data); i++ {
		if i+len(prefix) <= len(data) && string(data[i:i+len(prefix)]) == prefix {
			// Walk until digits start.
			j := i + len(prefix)
			for j < len(data) && (data[j] == ' ' || data[j] == '\t') {
				j++
			}
			k := j
			for k < len(data) && data[k] >= '0' && data[k] <= '9' {
				k++
			}
			if k > j {
				var kb int
				for _, c := range data[j:k] {
					kb = kb*10 + int(c-'0')
				}
				return (kb + (1 << 19)) >> 20 // KB → GB, rounded
			}
			return 0
		}
	}
	return 0
}
