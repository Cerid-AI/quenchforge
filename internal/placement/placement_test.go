// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package placement

import "testing"

func TestDevice(t *testing.T) {
	if GPU.String() != "gpu" || CPU.String() != "cpu" {
		t.Fatalf("String: gpu=%q cpu=%q", GPU.String(), CPU.String())
	}
	if GPU.GPULayers() != "999" || CPU.GPULayers() != "0" {
		t.Fatalf("GPULayers: gpu=%q cpu=%q", GPU.GPULayers(), CPU.GPULayers())
	}
}

func TestAMDDefaults(t *testing.T) {
	p := NewPolicy(true, nil)
	// chat + rerank are latency / single-request bound — CPU beats the AMD
	// Metal path on both. embed + code-embed are batched-throughput — GPU.
	for _, k := range []string{KindChat, KindRerank} {
		if p.Device(k) != CPU {
			t.Errorf("AMD %s should default to CPU, got %v", k, p.Device(k))
		}
	}
	for _, k := range []string{KindEmbed, KindCodeEmbed} {
		if p.Device(k) != GPU {
			t.Errorf("AMD %s should default to GPU, got %v", k, p.Device(k))
		}
	}
}

func TestNonAMDAllGPU(t *testing.T) {
	p := NewPolicy(false, nil)
	for _, k := range []string{KindChat, KindEmbed, KindCodeEmbed, KindRerank} {
		if p.Device(k) != GPU {
			t.Errorf("non-AMD %s should be GPU, got %v", k, p.Device(k))
		}
	}
}

func TestOverrides(t *testing.T) {
	p := NewPolicy(true, map[string]string{"embed": "cpu", "chat": "GPU", "rerank": "garbage", "code-embed": ""})
	if p.Device(KindEmbed) != CPU {
		t.Errorf("embed override to cpu failed: %v", p.Device(KindEmbed))
	}
	if p.Device(KindChat) != GPU {
		t.Errorf("chat override to gpu (case-insensitive) failed: %v", p.Device(KindChat))
	}
	if p.Device(KindRerank) != CPU {
		t.Errorf("invalid rerank override should be ignored (AMD default CPU): %v", p.Device(KindRerank))
	}
	if p.Device(KindCodeEmbed) != GPU {
		t.Errorf("empty code-embed override should be ignored: %v", p.Device(KindCodeEmbed))
	}
}

func TestRouteRequestAuto(t *testing.T) {
	p := NewPolicy(true, map[string]string{"embed": "auto"})
	// single/small -> CPU (latency), batched -> GPU (throughput)
	if d := p.RouteRequest(KindEmbed, 1, 4); d != CPU {
		t.Errorf("single embed under auto should route CPU, got %v", d)
	}
	if d := p.RouteRequest(KindEmbed, 32, 4); d != GPU {
		t.Errorf("batched embed under auto should route GPU, got %v", d)
	}
	if d := p.RouteRequest(KindEmbed, 4, 4); d != CPU {
		t.Errorf("at-threshold embed should route CPU, got %v", d)
	}
	// non-auto kind ignores batch shape
	if d := p.RouteRequest(KindChat, 999, 4); d != CPU {
		t.Errorf("fixed cpu chat should ignore batch, got %v", d)
	}
}
