// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin || linux

package supervisor

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// newProcAttr puts the child in its own process group so SIGTERM/SIGKILL
// can target the whole subtree (Setpgid=true means the kernel assigns a new
// pgid equal to the child's pid).
func newProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}

// signalGroup sends sig to the process group identified by pid (negative
// pid in kill(2) syntax). Returns the syscall error verbatim so callers
// can detect ESRCH ("no such process") and skip cleanup.
func signalGroup(pid int, sig syscall.Signal) error {
	if pid <= 0 {
		return fmt.Errorf("supervisor: invalid pid %d", pid)
	}
	return syscall.Kill(-pid, sig)
}

// processIdentity returns a running process's start time and executable
// path. The pair identifies a process across PID recycling: the kernel hands
// out PIDs again freely, but a recycled PID cannot share both the exact start
// timestamp and the executable path of the process we spawned.
//
// Uses `ps` (`lstart` and `comm`) — works on darwin and linux without /proc.
// Both values are opaque tokens; the reaper only ever compares them against
// what it recorded for the same PID at spawn time, never parses them.
func processIdentity(pid int) (start, execPath string, err error) {
	if start, err = psField(pid, "lstart="); err != nil {
		return "", "", err
	}
	if execPath, err = psField(pid, "comm="); err != nil {
		return "", "", err
	}
	if start == "" || execPath == "" {
		return "", "", fmt.Errorf("supervisor: empty ps identity for pid %d", pid)
	}
	return start, execPath, nil
}

func psField(pid int, field string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ps", "-o", field, "-p",
		fmt.Sprintf("%d", pid)).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
