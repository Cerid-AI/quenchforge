// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package discovery

import (
	"context"
	"errors"
	"testing"
)

func TestServiceWithDefaults(t *testing.T) {
	cases := []struct {
		name string
		in   Service
		want Service
	}{
		{
			"all empty",
			Service{Port: 11434},
			Service{
				Instance: "Quenchforge",
				Type:     "_quenchforge._tcp",
				Domain:   "local.",
				Port:     11434,
			},
		},
		{
			"keep overrides",
			Service{
				Instance:   "Custom",
				Type:       "_other._tcp",
				Domain:     "site.",
				Port:       8080,
				TXTRecords: []string{"x=1"},
			},
			Service{
				Instance:   "Custom",
				Type:       "_other._tcp",
				Domain:     "site.",
				Port:       8080,
				TXTRecords: []string{"x=1"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.in.withDefaults()
			if got.Instance != tc.want.Instance ||
				got.Type != tc.want.Type ||
				got.Domain != tc.want.Domain ||
				got.Port != tc.want.Port {
				t.Errorf("withDefaults:\n  got %+v\n want %+v", got, tc.want)
			}
		})
	}
}

func TestStartRequiresPort(t *testing.T) {
	_, err := Start(context.Background(), Service{Port: 0})
	if err == nil {
		t.Fatal("Start: expected error on missing port, got nil")
	}
	if !errors.Is(err, errMissingPort) {
		t.Errorf("Start: error = %v, want errMissingPort", err)
	}
}
