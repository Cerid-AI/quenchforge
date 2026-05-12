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

// commandLineForPID returns the argv joined by spaces for a running
// process. Uses `ps -o command= -p PID` — works on both darwin and
// linux without needing /proc.
func commandLineForPID(pid int) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ps", "-o", "command=", "-p",
		fmt.Sprintf("%d", pid)).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
