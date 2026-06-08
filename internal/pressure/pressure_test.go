// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package pressure

import "testing"

func TestForPlan(t *testing.T) {
	l := Limits{Max: 6, DisplayActive: 1, DisplayActiveDuty: 0.5}
	cases := []struct {
		name     string
		r        Reading
		wantConc int
		wantDuty float64
	}{
		{"headless normal → full, no gaps", Reading{DisplayActive: false, MemPressure: MemNormal}, 6, 1.0},
		{"display asleep treated headless", Reading{DisplayActive: false, MemPressure: MemNormal}, 6, 1.0},
		{"headless critical → serialize, no gaps", Reading{DisplayActive: false, MemPressure: MemCritical}, 1, 1.0},
		{"display active normal → conc1 + duty", Reading{DisplayActive: true, MemPressure: MemNormal}, 1, 0.5},
		{"display active warn → serialize + duty", Reading{DisplayActive: true, MemPressure: MemWarn}, 1, 0.5},
		{"display active critical → tighter duty", Reading{DisplayActive: true, MemPressure: MemCritical}, 1, 0.3},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := l.For(c.r)
			if p.Concurrency != c.wantConc || p.Duty != c.wantDuty {
				t.Errorf("For(%+v) = {conc=%d duty=%v}, want {conc=%d duty=%v}",
					c.r, p.Concurrency, p.Duty, c.wantConc, c.wantDuty)
			}
		})
	}
}

func TestForClamps(t *testing.T) {
	// Invalid Max → 1; invalid duty → 0.5 default.
	p := (Limits{Max: 0, DisplayActive: 0, DisplayActiveDuty: 0}).For(Reading{DisplayActive: true, MemPressure: MemNormal})
	if p.Concurrency != 1 || p.Duty != 0.5 {
		t.Errorf("clamp = {conc=%d duty=%v}, want {1, 0.5}", p.Concurrency, p.Duty)
	}
	// DisplayActive > Max → clamped to Max (headless uses Max here).
	p2 := (Limits{Max: 4, DisplayActive: 10, DisplayActiveDuty: 0.5}).For(Reading{DisplayActive: false, MemPressure: MemNormal})
	if p2.Concurrency != 4 {
		t.Errorf("headless conc = %d, want 4", p2.Concurrency)
	}
}

func TestSensorNeverPanics(t *testing.T) {
	r := NewSensor().Read()
	if r.MemPressure < 1 {
		t.Errorf("MemPressure = %d, want >= 1", r.MemPressure)
	}
}
