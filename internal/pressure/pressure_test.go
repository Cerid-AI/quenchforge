// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package pressure

import "testing"

func TestTarget(t *testing.T) {
	l := Limits{Max: 6, DisplayActive: 2}
	cases := []struct {
		name string
		r    Reading
		want int
	}{
		{"headless normal → full", Reading{DisplayActive: false, MemPressure: MemNormal}, 6},
		{"display asleep treated as headless", Reading{DisplayActive: false, MemPressure: MemNormal}, 6},
		{"display active normal → reserve headroom", Reading{DisplayActive: true, MemPressure: MemNormal}, 2},
		{"display active warn → halve (2→1)", Reading{DisplayActive: true, MemPressure: MemWarn}, 1},
		{"display active critical → 1", Reading{DisplayActive: true, MemPressure: MemCritical}, 1},
		{"headless critical → 1 (shed)", Reading{DisplayActive: false, MemPressure: MemCritical}, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := l.Target(c.r); got != c.want {
				t.Errorf("Target(%+v) = %d, want %d", c.r, got, c.want)
			}
		})
	}
}

func TestTargetHalvingRoundsUp(t *testing.T) {
	l := Limits{Max: 8, DisplayActive: 4}
	if got := l.Target(Reading{DisplayActive: true, MemPressure: MemWarn}); got != 2 {
		t.Errorf("warn halve of 4 = %d, want 2", got)
	}
}

func TestTargetClamps(t *testing.T) {
	// Max invalid → clamps to 1.
	if got := (Limits{Max: 0, DisplayActive: 0}).Target(Reading{DisplayActive: false, MemPressure: MemNormal}); got != 1 {
		t.Errorf("invalid Max headless = %d, want 1", got)
	}
	// DisplayActive > Max → clamped to Max.
	if got := (Limits{Max: 4, DisplayActive: 10}).Target(Reading{DisplayActive: true, MemPressure: MemNormal}); got != 4 {
		t.Errorf("DisplayActive>Max = %d, want 4 (clamped)", got)
	}
}

func TestSensorNeverPanics(t *testing.T) {
	// The platform sensor must always return a usable Reading.
	r := NewSensor().Read()
	if r.MemPressure < 1 {
		t.Errorf("MemPressure = %d, want >= 1", r.MemPressure)
	}
}
