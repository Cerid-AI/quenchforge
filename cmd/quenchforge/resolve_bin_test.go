// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import "testing"

func TestHomebrewBinsNativePrefixFirst(t *testing.T) {
	cases := map[string][]string{
		"arm64": {"/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"},
		"amd64": {"/usr/local/bin/llama-server", "/opt/homebrew/bin/llama-server"},
	}
	for goarch, want := range cases {
		got := homebrewBins("llama-server", goarch)
		if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
			t.Errorf("homebrewBins(llama-server, %s) = %v, want %v", goarch, got, want)
		}
	}
}
