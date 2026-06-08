// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin

package pressure

// NewSensor on non-darwin returns a sensor that always reports headless +
// normal memory, so the governor leaves the admission ceiling at Max (full
// throughput). Quenchforge only ships for darwin; this keeps `go build` and
// `go test ./...` green on the cross-platform CI runners.
func NewSensor() Sensor { return headlessSensor{} }

type headlessSensor struct{}

func (headlessSensor) Read() Reading {
	return Reading{DisplayActive: false, MemPressure: MemNormal}
}
