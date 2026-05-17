// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"testing"
	"time"
)

func TestSampleStore_RingBufferCap(t *testing.T) {
	s := newSampleStore(5)
	for i := range 12 {
		s.add(time.Duration(i+1) * time.Millisecond)
	}
	if len(s.samples) != 5 {
		t.Errorf("sample store grew past cap: got %d, want 5", len(s.samples))
	}
	// Newest entries retained; oldest dropped.
	first := s.samples[0]
	last := s.samples[len(s.samples)-1]
	if first != 8*time.Millisecond {
		t.Errorf("oldest in-window sample = %v, want 8ms", first)
	}
	if last != 12*time.Millisecond {
		t.Errorf("newest sample = %v, want 12ms", last)
	}
}

func TestSampleStore_PercentilesOnSortedData(t *testing.T) {
	s := newSampleStore(100)
	for i := 1; i <= 100; i++ {
		s.add(time.Duration(i) * time.Millisecond)
	}
	p50, p99 := s.percentiles()
	if p50 < 48 || p50 > 52 {
		t.Errorf("p50 = %.1f, want ~50ms", p50)
	}
	if p99 < 98 || p99 > 100 {
		t.Errorf("p99 = %.1f, want ~99ms", p99)
	}
}

func TestSampleStore_EmptyReturnsZero(t *testing.T) {
	s := newSampleStore(10)
	p50, p99 := s.percentiles()
	if p50 != 0 || p99 != 0 {
		t.Errorf("empty store percentiles = %.1f / %.1f, want 0 / 0", p50, p99)
	}
}

func TestSafeRatio(t *testing.T) {
	if got := safeRatio(10, 2); got != 5 {
		t.Errorf("safeRatio(10, 2) = %.1f, want 5", got)
	}
	if got := safeRatio(10, 0); got != 0 {
		t.Errorf("safeRatio divide-by-zero = %.1f, want 0", got)
	}
	if got := safeRatio(0, 0); got != 0 {
		t.Errorf("safeRatio(0, 0) = %.1f, want 0", got)
	}
}

func TestRunStats_TracksTotalAndErrors(t *testing.T) {
	s := newRunStats()
	for i := range 20 {
		s.recordSample(i%5 == 0)
	}
	if s.total != 20 {
		t.Errorf("total = %d, want 20", s.total)
	}
	if s.errs != 4 {
		t.Errorf("errs = %d, want 4 (every 5th sample)", s.errs)
	}
}

func TestBuildEmbedBody_HandlesBatchSize(t *testing.T) {
	body := buildEmbedBody("nomic-embed-text-v1.5", 3)
	// Body is JSON. Cheap shape check: it contains the model and at
	// least three sample-text fingerprints (token "Metal" appears in
	// the first sample; "Quenchforge" in the second; "Sustained" in
	// the third).
	cases := []string{
		`"model":"nomic-embed-text-v1.5"`,
		"Metal",
		"Quenchforge",
		"Sustained",
	}
	s := string(body)
	for _, c := range cases {
		if !contains(s, c) {
			t.Errorf("buildEmbedBody body missing fingerprint %q\nbody=%s", c, s)
		}
	}
}

func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
