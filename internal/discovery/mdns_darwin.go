// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

package discovery

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
)

var errMissingPort = errors.New("discovery: Service.Port is required")

// platformStart spawns `dns-sd -R "<instance>" <type> <domain> <port> [TXT...]`
// and returns an Advertiser that holds the child process. The child runs for
// the lifetime of the advertisement; killing it withdraws the record.
func platformStart(ctx context.Context, s Service) (Advertiser, error) {
	args := []string{
		"-R",
		s.Instance,
		s.Type,
		s.Domain,
		fmt.Sprintf("%d", s.Port),
	}
	args = append(args, s.TXTRecords...)

	// Wrap with our own context so Stop can cancel even if the caller's ctx
	// outlives the advertisement.
	advCtx, cancel := context.WithCancel(ctx)
	cmd := exec.CommandContext(advCtx, "/usr/bin/dns-sd", args...)
	// dns-sd is chatty on stdout; redirect to /dev/null. Stderr stays on the
	// parent process so registration failures surface in `quenchforge serve`'s
	// log.
	cmd.Stdout = io.Discard

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("discovery: spawn dns-sd: %w", err)
	}

	adv := &dnsSDAdvertiser{cmd: cmd, cancel: cancel}
	adv.running.Store(true)

	// Reap the child if it exits on its own — that flips Running() to false
	// and lets the supervising goroutine notice via select.
	go func() {
		_ = cmd.Wait()
		adv.running.Store(false)
		cancel()
	}()

	return adv, nil
}

// dnsSDAdvertiser wraps the dns-sd child.
type dnsSDAdvertiser struct {
	cmd     *exec.Cmd
	cancel  context.CancelFunc
	running atomic.Bool

	stopOnce sync.Once
	stopErr  error
}

func (a *dnsSDAdvertiser) Stop() error {
	a.stopOnce.Do(func() {
		a.running.Store(false)
		a.cancel()
		// CommandContext+cancel sends SIGKILL; Wait returns whatever the
		// child reported. We don't surface a wait error here because killing
		// our own child intentionally always exits with a signal status.
		_ = a.cmd.Wait()
	})
	return a.stopErr
}

func (a *dnsSDAdvertiser) Running() bool { return a.running.Load() }
