// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin && !linux

// Stub for platforms quenchforge doesn't actually ship to. Kept compilable
// so `go build ./...` works on those platforms in case someone tries.

package supervisor

import "syscall"

func newProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{}
}

func signalGroup(pid int, sig syscall.Signal) error {
	return syscall.ENOSYS
}

func commandLineForPID(pid int) (string, error) {
	return "", syscall.ENOSYS
}
