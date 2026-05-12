// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestPriorityStringer(t *testing.T) {
	cases := map[Priority]string{
		PriorityChat:       "chat",
		PriorityEmbed:      "embed",
		PriorityRerank:     "rerank",
		PriorityBackground: "background",
		Priority(42):       "priority(42)",
		Priority(-3):       "priority(-3)",
	}
	for p, want := range cases {
		if got := p.String(); got != want {
			t.Errorf("Priority(%d).String() = %q, want %q", p, got, want)
		}
	}
}

func TestSingleSlotPicksHighestPriority(t *testing.T) {
	s := New(1)

	// First, grab the only slot for a low-priority workload.
	low, err := s.Acquire(context.Background(), PriorityBackground)
	if err != nil {
		t.Fatalf("Acquire low: %v", err)
	}

	// Queue a chat and an embed; chat should be admitted first when low releases.
	type out struct {
		who string
		at  time.Time
	}
	results := make(chan out, 2)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		release, err := s.Acquire(context.Background(), PriorityEmbed)
		if err != nil {
			t.Errorf("Acquire embed: %v", err)
			return
		}
		results <- out{"embed", time.Now()}
		// Hold briefly so the chat slot has to wait its turn.
		time.Sleep(15 * time.Millisecond)
		release()
	}()
	// Queue the chat slot AFTER the embed so they have stable insertion order.
	time.Sleep(5 * time.Millisecond)
	go func() {
		defer wg.Done()
		release, err := s.Acquire(context.Background(), PriorityChat)
		if err != nil {
			t.Errorf("Acquire chat: %v", err)
			return
		}
		results <- out{"chat", time.Now()}
		release()
	}()
	time.Sleep(5 * time.Millisecond)

	// Now release the low-priority lock — chat should win the race against embed.
	low()
	wg.Wait()
	close(results)

	var got []out
	for r := range results {
		got = append(got, r)
	}
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	if got[0].who != "chat" {
		t.Errorf("first admitted = %q, want chat", got[0].who)
	}
}

func TestFIFOWithinSamePriority(t *testing.T) {
	s := New(1)
	hold, _ := s.Acquire(context.Background(), PriorityChat)

	const n = 4
	var order []int
	var mu sync.Mutex
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r, err := s.Acquire(context.Background(), PriorityEmbed)
			if err != nil {
				t.Errorf("Acquire #%d: %v", i, err)
				return
			}
			mu.Lock()
			order = append(order, i)
			mu.Unlock()
			r()
		}(i)
		// Give the goroutine a chance to enqueue in order.
		time.Sleep(2 * time.Millisecond)
	}
	time.Sleep(20 * time.Millisecond)
	hold()
	wg.Wait()

	for i := 0; i < n; i++ {
		if order[i] != i {
			t.Errorf("admission order = %v, want sequential", order)
			break
		}
	}
}

func TestConcurrencyN(t *testing.T) {
	const cap = 3
	s := New(cap)
	var inFlight atomic.Int32
	var peak atomic.Int32

	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, _ := s.Acquire(context.Background(), PriorityChat)
			now := inFlight.Add(1)
			for {
				p := peak.Load()
				if now <= p || peak.CompareAndSwap(p, now) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			r()
		}()
	}
	wg.Wait()
	if int(peak.Load()) > cap {
		t.Errorf("peak in-flight = %d, want <= %d", peak.Load(), cap)
	}
	if int(peak.Load()) < 2 {
		t.Errorf("peak in-flight = %d; concurrency seems broken", peak.Load())
	}
}

func TestCancelDropsFromQueue(t *testing.T) {
	s := New(1)
	hold, _ := s.Acquire(context.Background(), PriorityChat)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := s.Acquire(ctx, PriorityChat)
		done <- err
	}()

	// Make sure the request is pending.
	time.Sleep(10 * time.Millisecond)
	if got := s.Pending(); got != 1 {
		t.Fatalf("Pending = %d, want 1", got)
	}
	cancel()
	if err := <-done; err == nil {
		t.Error("Acquire after cancel: nil error, want ctx.Err")
	}
	hold()

	// Pending should drain (the maybeAdmit pass skips cancelled requests).
	if got := s.Pending(); got != 0 {
		t.Errorf("Pending after cancel = %d, want 0", got)
	}
}

func TestActiveAndPendingCounters(t *testing.T) {
	s := New(2)
	if s.Active() != 0 || s.Pending() != 0 {
		t.Fatalf("fresh scheduler: active=%d pending=%d", s.Active(), s.Pending())
	}
	r1, _ := s.Acquire(context.Background(), PriorityChat)
	r2, _ := s.Acquire(context.Background(), PriorityChat)
	if s.Active() != 2 {
		t.Errorf("Active after 2 admits = %d, want 2", s.Active())
	}
	// Third request queues
	done := make(chan func(), 1)
	go func() {
		r, _ := s.Acquire(context.Background(), PriorityChat)
		done <- r
	}()
	time.Sleep(5 * time.Millisecond)
	if s.Pending() != 1 {
		t.Errorf("Pending with 3rd waiting = %d, want 1", s.Pending())
	}
	r1()
	r := <-done
	r()
	r2()
	if s.Active() != 0 {
		t.Errorf("Active after all released = %d, want 0", s.Active())
	}
}
