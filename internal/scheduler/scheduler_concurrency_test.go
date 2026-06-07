// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package scheduler

import (
	"context"
	"testing"
	"time"
)

func waitFor(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for: %s", msg)
}

func TestSetConcurrencyRaiseAdmitsPending(t *testing.T) {
	s := New(1)
	r1, _ := s.Acquire(context.Background(), PriorityChat) // active=1

	got := make(chan func(), 2)
	for i := 0; i < 2; i++ {
		go func() {
			r, _ := s.Acquire(context.Background(), PriorityEmbed)
			got <- r
		}()
	}
	waitFor(t, func() bool { return s.Pending() == 2 }, "two pending requests")

	s.SetConcurrency(3) // should admit both now-fitting requests
	r2 := <-got
	r3 := <-got
	if s.Active() != 3 {
		t.Fatalf("Active after raise = %d, want 3", s.Active())
	}
	if s.Concurrency() != 3 {
		t.Fatalf("Concurrency = %d, want 3", s.Concurrency())
	}
	r1()
	r2()
	r3()
}

func TestSetConcurrencyLowerDefersToDrain(t *testing.T) {
	s := New(3)
	r1, _ := s.Acquire(context.Background(), PriorityChat)
	r2, _ := s.Acquire(context.Background(), PriorityChat)
	r3, _ := s.Acquire(context.Background(), PriorityChat) // active=3

	s.SetConcurrency(1) // must NOT evict in-flight work
	if s.Active() != 3 {
		t.Fatalf("Active after lowering = %d, want 3 (no eviction)", s.Active())
	}

	got := make(chan func(), 1)
	go func() {
		r, _ := s.Acquire(context.Background(), PriorityChat)
		got <- r
	}()
	waitFor(t, func() bool { return s.Pending() == 1 }, "the new request queues")

	r1() // active 3->2; 2 >= ceiling(1) so still no admit
	time.Sleep(15 * time.Millisecond)
	select {
	case <-got:
		t.Fatal("admitted at active=2 while ceiling=1")
	default:
	}

	r2() // active 2->1; still no admit
	r3() // active 1->0; now < ceiling, admit the waiter
	r4 := <-got
	r4()
}

func TestSetConcurrencyClampsToOne(t *testing.T) {
	s := New(2)
	s.SetConcurrency(0)
	if s.Concurrency() != 1 {
		t.Fatalf("Concurrency after SetConcurrency(0) = %d, want 1 (clamped)", s.Concurrency())
	}
}
