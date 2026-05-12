// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package hardware

import (
	"context"
	"time"
)

// commandTimeout returns a context capped at d for shell-out probes. Extracted
// so detect_darwin.go and tests share the same cancellation behavior.
func commandTimeout(d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), d)
}
