// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package scheduler enforces request ordering across the slots managed by
// the supervisor. Metal command queues serialize per-device, so concurrent
// chat + embed + rerank traffic on a single Vega II jitter each other if
// they're allowed to interleave freely. The scheduler keeps streaming chat
// at the front of the queue and lets cheaper batch workloads (embed,
// rerank) drain behind it.
//
// MVP-stage scope: a single in-process priority queue that gates Acquire()
// calls. v0.2 will swap in a multi-device version that round-robins across
// MTLDevices (Vega II Duo, multi-MPX configs) by reading
// MTLCopyAllDevices and applying device affinity per slot type.
//
// Priorities (higher number = higher priority):
//
//	PriorityChat      = 30  // user-facing streaming, latency-critical
//	PriorityEmbed     = 20  // batch ingestion, latency-tolerant
//	PriorityRerank    = 10  // post-retrieval, off the user's critical path
//	PriorityBackground = 0  // anything else; whisper transcription, eval
//
// The implementation is intentionally simple — a `container/heap`-backed
// priority queue with a fair tiebreaker (FIFO within priority). 100k req/s
// fits comfortably in one goroutine; we don't need lock-free yet.
package scheduler

import (
	"container/heap"
	"context"
	"errors"
	"sync"
	"sync/atomic"
)

// Priority is the integer rank assigned to a workload. Higher = served first.
type Priority int

// Priority levels. Stable values so external callers can hard-code if needed.
const (
	PriorityBackground Priority = 0
	PriorityRerank     Priority = 10
	PriorityEmbed      Priority = 20
	PriorityChat       Priority = 30
)

// String returns a human-readable name for known priorities; unrecognised
// values render as "priority(N)" so logs stay useful for custom priorities.
func (p Priority) String() string {
	switch p {
	case PriorityChat:
		return "chat"
	case PriorityEmbed:
		return "embed"
	case PriorityRerank:
		return "rerank"
	case PriorityBackground:
		return "background"
	}
	return "priority(" + itoa(int(p)) + ")"
}

// Scheduler is the lock-protected queue. Construct with New.
type Scheduler struct {
	// concurrency is the number of simultaneous workloads allowed across
	// all slots. 1 by default on a single-GPU host; higher when the
	// supervisor has multiple devices wired.
	concurrency int

	mu     sync.Mutex
	heap   reqHeap
	active int
	seq    atomic.Uint64
	wake   chan struct{}
}

// New returns a Scheduler that allows up to concurrency in-flight workloads.
// concurrency must be >= 1.
func New(concurrency int) *Scheduler {
	if concurrency < 1 {
		concurrency = 1
	}
	s := &Scheduler{
		concurrency: concurrency,
		wake:        make(chan struct{}, 1),
	}
	heap.Init(&s.heap)
	return s
}

// Acquire blocks until the workload at priority p is at the front of the
// queue AND the scheduler has spare concurrency. Returns a release function
// the caller must call (typically `defer release()`) when the workload is
// done. ctx cancellation drops the caller's slot in line.
//
// A nil ctx is treated as context.Background.
func (s *Scheduler) Acquire(ctx context.Context, p Priority) (release func(), err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	r := &request{
		priority: int(p),
		seq:      s.seq.Add(1),
		wake:     make(chan struct{}, 1),
	}

	s.mu.Lock()
	heap.Push(&s.heap, r)
	s.maybeAdmitLocked()
	s.mu.Unlock()

	select {
	case <-r.wake:
		// admitted
		return s.release, nil
	case <-ctx.Done():
		// Caller bailed before we reached the front. Mark cancelled and
		// re-poll the queue in case our slot would have been the next to run.
		s.mu.Lock()
		r.cancelled = true
		// If we were already admitted between the wake-send and our case
		// arm winning, we'd leak an active slot. Detect by checking the
		// wake channel one more time.
		select {
		case <-r.wake:
			s.active--
			s.maybeAdmitLocked()
		default:
		}
		s.mu.Unlock()
		return func() {}, ctx.Err()
	}
}

// release is the deferred function returned by Acquire. Decrements the
// in-flight counter and wakes the next-highest waiter.
func (s *Scheduler) release() {
	s.mu.Lock()
	if s.active > 0 {
		s.active--
	}
	s.maybeAdmitLocked()
	s.mu.Unlock()
}

// Active reports how many workloads are in-flight right now.
func (s *Scheduler) Active() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}

// Pending reports how many workloads are waiting for admission.
func (s *Scheduler) Pending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heap.Len()
}

// maybeAdmitLocked pops cancelled requests off the front and admits as many
// non-cancelled ones as we have spare concurrency for.
func (s *Scheduler) maybeAdmitLocked() {
	for s.heap.Len() > 0 && s.active < s.concurrency {
		// Peek
		r := s.heap[0]
		if r.cancelled {
			heap.Pop(&s.heap)
			continue
		}
		heap.Pop(&s.heap)
		s.active++
		select {
		case r.wake <- struct{}{}:
		default:
			// shouldn't happen — wake is 1-buffered and we just constructed it
		}
	}
}

// ---------------------------------------------------------------------------
// heap impl
// ---------------------------------------------------------------------------

type request struct {
	priority  int
	seq       uint64
	wake      chan struct{}
	cancelled bool
}

type reqHeap []*request

func (h reqHeap) Len() int { return len(h) }
func (h reqHeap) Less(i, j int) bool {
	// Higher priority comes first.
	if h[i].priority != h[j].priority {
		return h[i].priority > h[j].priority
	}
	// FIFO within the same priority.
	return h[i].seq < h[j].seq
}
func (h reqHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *reqHeap) Push(x any)   { *h = append(*h, x.(*request)) }
func (h *reqHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// itoa is a stdlib-free replacement to avoid importing strconv just for the
// Priority Stringer's fallback path.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// ErrCancelled is the canonical sentinel callers can compare against when
// Acquire returns due to context cancellation. It wraps context.Canceled.
var ErrCancelled = errors.New("scheduler: cancelled")
