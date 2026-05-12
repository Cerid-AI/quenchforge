// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build !darwin

package discovery

import (
	"context"
	"errors"
)

var errMissingPort = errors.New("discovery: Service.Port is required")

// platformStart on non-Darwin returns a no-op Advertiser. Quenchforge is
// macOS-only by design; the stub keeps `go build ./...` green on Linux CI.
func platformStart(_ context.Context, _ Service) (Advertiser, error) {
	return noopAdvertiser{}, nil
}

type noopAdvertiser struct{}

func (noopAdvertiser) Stop() error   { return nil }
func (noopAdvertiser) Running() bool { return false }
