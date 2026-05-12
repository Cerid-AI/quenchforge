// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package discovery advertises the quenchforge gateway on the local network
// via mDNS / Bonjour so other LAN clients (Cerid AI, other quenchforge
// peers, generic Ollama clients with mDNS support) can find it without
// hard-coded URLs.
//
// On macOS the advertisement runs through the system's `mDNSResponder`
// daemon by spawning `dns-sd -R`. That choice is deliberate:
//
//  1. mDNSResponder is the canonical TCC-permission boundary on Sonoma+.
//     Operators see the documented "Quenchforge would like to find and
//     connect to devices on your local network" prompt once, owned by
//     Apple's daemon, not us — exactly what the cerid-ai integration plan
//     asked for.
//  2. Zero external Go dependencies. The next-best option (grandcat/zeroconf
//     or hashicorp/mdns) would add ~10 MB of transitive vendoring.
//  3. mDNSResponder handles interface enumeration, IPv4 + IPv6 dual-stack,
//     and link-change events automatically. Re-implementing that correctly
//     in Go is at-best the same code Apple already ships.
//
// On non-Darwin platforms the API is the same but returns immediately with
// no advertisement — Quenchforge is macOS-only, and the stub keeps
// `go build ./...` green on Linux CI.
//
// Advertisement defaults to OFF. Operators enable via the
// QUENCHFORGE_ADVERTISE_MDNS=true env var (read at `quenchforge serve`
// startup). When enabled, withdrawal happens automatically when the
// serve context is cancelled.
package discovery

import "context"

// Service describes one Bonjour service registration.
type Service struct {
	// Instance is the user-visible service name, e.g. "Quenchforge on
	// MyMac". Goes in the Bonjour browse list. Defaults to "Quenchforge"
	// if empty.
	Instance string

	// Type is the service-type triplet, e.g. "_quenchforge._tcp". Must
	// include the protocol suffix. Defaults to "_quenchforge._tcp".
	Type string

	// Domain is the Bonjour domain. Defaults to "local." for link-local.
	Domain string

	// Port is the TCP port the service listens on. Required.
	Port int

	// TXTRecords are key=value strings attached to the SRV record. Clients
	// read these to discover protocol versions, supported routes, etc.
	// Example: []string{"version=0.1.0", "api=ollama,openai"}.
	TXTRecords []string
}

// withDefaults returns a copy of s with empty fields filled in.
func (s Service) withDefaults() Service {
	if s.Instance == "" {
		s.Instance = "Quenchforge"
	}
	if s.Type == "" {
		s.Type = "_quenchforge._tcp"
	}
	if s.Domain == "" {
		s.Domain = "local."
	}
	return s
}

// Advertiser is the abstract handle returned by Start. Calling Stop is
// safe and idempotent — the second call is a no-op.
type Advertiser interface {
	Stop() error
	// Running reports whether the advertisement is still active. Returns
	// false after Stop or after the underlying daemon exits.
	Running() bool
}

// Start advertises s on the local network. The advertisement remains active
// until the caller calls Stop or ctx is cancelled. Returns an error if the
// platform doesn't support mDNS or if the system daemon can't be spawned.
//
// The returned Advertiser is safe for concurrent Stop calls.
func Start(ctx context.Context, s Service) (Advertiser, error) {
	if s.Port == 0 {
		return nil, errMissingPort
	}
	return platformStart(ctx, s.withDefaults())
}
