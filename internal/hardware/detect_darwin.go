// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin

// Hardware probe on macOS. Uses sysctlbyname for CPU/RAM (cheap, no
// frameworks linked) and a sh-out to `system_profiler SPDisplaysDataType`
// for GPU enumeration (no CGo, no Metal framework linkage required at
// build time). When a future v0.2 needs real-time Metal device events
// — recommendedMaxWorkingSetSize, MTLDeviceWasAddedNotification — we'll
// switch to CGo + Metal.framework here, behind a build tag.
//
// The shell-out approach is deliberate for MVP:
//   - Works on minimal CI images (no Metal headers required at build).
//   - Avoids CGo at this layer entirely — keeps cross-compilation easy.
//   - system_profiler is bundled in every macOS install since 10.0.
//
// Trade-off: ~200-500ms extra at first launch vs. a direct Metal API call.
// Acceptable for a once-per-boot probe; cached for subsequent reads.
package hardware

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func detect() (Info, error) {
	info := Info{}

	// CPU + RAM via `sysctl -n` shell-out. The Go syscall package on darwin
	// only exposes the string variant; numeric sysctl values would need an
	// `golang.org/x/sys/unix` dependency we'd rather not pull in for two
	// uses. Cost is ~5ms per call — negligible at startup.
	info.CPU = sysctlNString("machdep.cpu.brand_string")
	if v, err := strconv.Atoi(sysctlNString("hw.logicalcpu")); err == nil {
		info.CPUCores = v
	}
	if v, err := strconv.ParseUint(sysctlNString("hw.memsize"), 10, 64); err == nil {
		info.TotalRAMGB = int((v + (1 << 29)) >> 30) // round to nearest GB
	}

	// OS version
	info.OSVersion = darwinOSVersion()

	// GPU via system_profiler (JSON output)
	devices, err := readDisplayDevices()
	if err != nil {
		// Detection failed — surface as "unknown" so callers can refuse to
		// start rather than silently picking the wrong profile.
		return Info{
			Profile:    ProfileUnknown,
			CPU:        info.CPU,
			CPUCores:   info.CPUCores,
			TotalRAMGB: info.TotalRAMGB,
			OSVersion:  info.OSVersion,
		}, fmt.Errorf("hardware: GPU enumeration failed: %w", err)
	}
	info.Devices = devices

	// Pick the headline GPU = highest-VRAM non-low-power device.
	for _, d := range devices {
		if d.LowPower {
			continue
		}
		if d.VRAMGB >= info.GPUVRAMGB {
			info.GPU = d.Name
			info.GPUVRAMGB = d.VRAMGB
		}
		info.HasMetal = true
	}
	// If only low-power devices are present (Intel iGPU on a 2018 MBP, etc.)
	// surface that one — Metal still works but we route to ProfileIGPU.
	if info.GPU == "" && len(devices) > 0 {
		info.GPU = devices[0].Name
		info.GPUVRAMGB = devices[0].VRAMGB
		info.HasMetal = true
	}

	info.Profile = classifyProfile(info)
	return info, nil
}

// classifyProfile maps the raw enumeration to a Profile constant. Order
// matters: more specific buckets first.
func classifyProfile(i Info) Profile {
	for _, d := range i.Devices {
		if d.AppleSilicon {
			return ProfileAppleSilicon
		}
	}
	gpu := strings.ToLower(i.GPU)
	switch {
	case strings.Contains(gpu, "vega ii") || strings.Contains(gpu, "vega 2"):
		return ProfileVegaPro
	case strings.Contains(gpu, "w6800x"):
		return ProfileW6800X
	case strings.Contains(gpu, "rx 5500") || strings.Contains(gpu, "rx 5700") ||
		strings.Contains(gpu, "5500m") || strings.Contains(gpu, "5700"):
		return ProfileRDNA1
	case strings.Contains(gpu, "rx 6700") || strings.Contains(gpu, "6700m") ||
		strings.Contains(gpu, "w6600") || strings.Contains(gpu, "w6800"):
		return ProfileRDNA2
	case strings.Contains(gpu, "iris") || strings.Contains(gpu, "intel hd") ||
		strings.Contains(gpu, "intel uhd"):
		return ProfileIGPU
	case i.HasMetal:
		// Discrete AMD on Intel Mac that didn't match any specific bucket —
		// fall through to vega-pro tuning as the closest defaults. Operators
		// hitting this should file a hardware_profile.yml issue.
		return ProfileVegaPro
	}
	return ProfileCPU
}

// ---------------------------------------------------------------------------
// sysctl helpers — shell out to /usr/sbin/sysctl -n. The Go syscall.Sysctl
// only returns strings, and golang.org/x/sys/unix.SysctlUint64 would pull
// in a dependency we use exactly twice. ~5ms latency at startup, fine.
// ---------------------------------------------------------------------------

// sysctlNString returns the raw `sysctl -n <name>` output, trimmed. Empty
// string on error so callers can use the zero-value path.
func sysctlNString(name string) string {
	ctx, cancel := commandTimeout(2 * time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "/usr/sbin/sysctl", "-n", name).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func darwinOSVersion() string {
	// `sw_vers -productVersion` is the canonical surface and is friendlier
	// than parsing kern.osproductversion (which sometimes lies about the
	// minor revision in beta builds).
	ctx, cancel := commandTimeout(3 * time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "/usr/bin/sw_vers", "-productVersion").Output()
	if err != nil {
		return ""
	}
	v := strings.TrimSpace(string(out))
	if v == "" {
		return ""
	}
	return "macOS " + v
}

// ---------------------------------------------------------------------------
// system_profiler JSON parsing
// ---------------------------------------------------------------------------

// readDisplayDevices invokes `system_profiler SPDisplaysDataType -json` and
// parses the result. The output schema is unusually-shaped:
//
//	{
//	  "SPDisplaysDataType": [
//	    {
//	      "sppci_model": "AMD Radeon Pro Vega II",
//	      "spdisplays_vram": "32 GB",
//	      ... or sometimes spdisplays_vram_shared for iGPUs ...
//	    },
//	    ...
//	  ]
//	}
//
// We tolerate missing keys and report what we can.
func readDisplayDevices() ([]Device, error) {
	ctx, cancel := commandTimeout(15 * time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "/usr/sbin/system_profiler",
		"-json", "SPDisplaysDataType").Output()
	if err != nil {
		return nil, fmt.Errorf("system_profiler: %w", err)
	}

	var payload struct {
		SPDisplays []map[string]any `json:"SPDisplaysDataType"`
	}
	if err := json.Unmarshal(out, &payload); err != nil {
		return nil, fmt.Errorf("parse system_profiler JSON: %w", err)
	}

	out2 := make([]Device, 0, len(payload.SPDisplays))
	for _, entry := range payload.SPDisplays {
		dev := Device{
			Name: stringOr(entry, "sppci_model", "Unknown GPU"),
		}
		// VRAM key varies: dedicated GPUs report `spdisplays_vram`, iGPUs
		// report `spdisplays_vram_shared`. Apple Silicon reports both;
		// pick the larger to capture unified memory.
		dev.VRAMGB = parseVRAMGB(entry, "spdisplays_vram", "spdisplays_vram_shared")

		// Detect Apple Silicon: vendor reads "sppci_vendor_apple", or the
		// model contains "Apple M".
		vendor := strings.ToLower(stringOr(entry, "sppci_vendor", ""))
		if strings.Contains(vendor, "apple") ||
			strings.Contains(strings.ToLower(dev.Name), "apple m") {
			dev.AppleSilicon = true
		}

		// Low-power = integrated. Heuristic: name contains Iris/Intel HD/UHD
		// or the only VRAM key was the shared variant.
		nameLow := strings.ToLower(dev.Name)
		if strings.Contains(nameLow, "iris") ||
			strings.Contains(nameLow, "intel hd") ||
			strings.Contains(nameLow, "intel uhd") {
			dev.LowPower = true
		}

		out2 = append(out2, dev)
	}
	return out2, nil
}

func stringOr(m map[string]any, key, fallback string) string {
	if v, ok := m[key].(string); ok && v != "" {
		return v
	}
	return fallback
}

// parseVRAMGB looks at one or more candidate keys, picks the largest value,
// and returns it rounded to whole GB. "8 GB" → 8, "32768 MB" → 32, "0 MB" → 0.
func parseVRAMGB(m map[string]any, keys ...string) int {
	var best int
	for _, k := range keys {
		s, ok := m[k].(string)
		if !ok || s == "" {
			continue
		}
		gb := parseSizeString(s)
		if gb > best {
			best = gb
		}
	}
	return best
}

func parseSizeString(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	// Split numeric portion from unit suffix.
	var numEnd int
	for numEnd < len(s) {
		c := s[numEnd]
		if c >= '0' && c <= '9' {
			numEnd++
			continue
		}
		break
	}
	if numEnd == 0 {
		return 0
	}
	n, err := strconv.Atoi(s[:numEnd])
	if err != nil {
		return 0
	}
	unit := strings.ToLower(strings.TrimSpace(s[numEnd:]))
	switch {
	case strings.HasPrefix(unit, "gb"), strings.HasPrefix(unit, "gib"):
		return n
	case strings.HasPrefix(unit, "mb"), strings.HasPrefix(unit, "mib"):
		return n / 1024
	}
	// Unknown unit — assume MB (system_profiler legacy).
	return n / 1024
}
