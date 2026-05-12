// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestExtractPlistVersion(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want int
	}{
		{
			"sonoma",
			`<dict><key>ProductVersion</key><string>14.5</string></dict>`,
			14,
		},
		{
			"sequoia",
			`<dict><key>ProductName</key><string>macOS</string>` +
				`<key>ProductVersion</key><string>15.0.1</string></dict>`,
			15,
		},
		{
			"tahoe",
			`<dict><key>ProductVersion</key><string>26.5</string></dict>`,
			26,
		},
		{
			"missing key",
			`<dict><key>SomethingElse</key><string>14.5</string></dict>`,
			0,
		},
		{
			"malformed",
			`<key>ProductVersion</key><string>not a number</string>`,
			0,
		},
		{
			"empty",
			``,
			0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := extractPlistVersion([]byte(tc.in))
			if got != tc.want {
				t.Errorf("extractPlistVersion(%q) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
}
