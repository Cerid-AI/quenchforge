// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package pressure

import "testing"

func TestExtractInt(t *testing.T) {
	// Real ioreg rendering of the IODisplayWrangler power dict.
	s := `"IOPowerManagement" = {"CapabilityFlags"=32832,"MaxPowerState"=4,"DevicePowerState"=4,"CurrentPowerState"=4}`
	if v, ok := extractInt(s, `"CurrentPowerState"=`); !ok || v != 4 {
		t.Errorf("CurrentPowerState = %d ok=%v, want 4 true", v, ok)
	}
	if v, ok := extractInt(s, `"MaxPowerState"=`); !ok || v != 4 {
		t.Errorf("MaxPowerState = %d ok=%v, want 4 true", v, ok)
	}
	if _, ok := extractInt(s, `"Missing"=`); ok {
		t.Error("extractInt found a missing key")
	}
}

func TestExtractIntAsleep(t *testing.T) {
	// Display asleep: CurrentPowerState drops below Max.
	s := `{"MaxPowerState"=4,"CurrentPowerState"=1}`
	cur, _ := extractInt(s, `"CurrentPowerState"=`)
	max, _ := extractInt(s, `"MaxPowerState"=`)
	if cur >= max {
		t.Errorf("asleep display should read cur(%d) < max(%d)", cur, max)
	}
}
