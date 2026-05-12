// Command quenchforge-preflight gates a Quenchforge install on supported
// hardware. Intended to ship behind a `curl -fsSL get.quenchforge.dev/preflight
// | sh` style preamble so users on unsupported configurations bounce off
// before downloading a binary that won't help them.
//
// Exit codes:
//
//	0 — supported (proceed with install)
//	1 — generic error (probe failure, bug in preflight)
//	2 — unsupported platform (not macOS, or macOS too old)
//	3 — unsupported hardware (no Metal, no discrete AMD or Apple Silicon)
//	4 — insufficient disk space for a baseline model + binary
//
// Output is machine-readable on stdout (one KEY=VALUE line per check) plus
// a human summary on stderr. Pipe stdout into a `source` or `eval` in shell
// installers.
package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"syscall"

	"github.com/cerid-ai/quenchforge/internal/hardware"
)

// Minimum macOS version. The plan calls for Sonoma 14.0 (Metal 3 + argument
// buffers tier 2). Anything earlier lacks features quenchforge relies on.
const minMacOSMajor = 14

// Minimum free disk (GB) on the install volume. Holds the universal binary
// (~50 MB), one decent-sized GGUF (~5 GB for a 7B q4), and headroom.
const minFreeDiskGB = 10

func main() {
	exitCode := 0
	report := map[string]string{}

	defer func() {
		// Always emit a machine-readable report before exiting so install
		// scripts can parse it without parsing the human-readable summary.
		for k, v := range report {
			fmt.Printf("%s=%s\n", k, v)
		}
		os.Exit(exitCode)
	}()

	// Phase 1: platform
	report["os"] = runtime.GOOS
	report["arch"] = runtime.GOARCH

	if runtime.GOOS != "darwin" {
		report["status"] = "unsupported-platform"
		report["reason"] = "quenchforge is macOS-only"
		fmt.Fprintln(os.Stderr, "[FAIL] Unsupported platform:", runtime.GOOS)
		fmt.Fprintln(os.Stderr, "       Quenchforge is macOS-only by design.")
		exitCode = 2
		return
	}

	// Phase 2: macOS version
	major, err := macOSMajor()
	if err == nil {
		report["macos_major"] = strconv.Itoa(major)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "[WARN] Could not determine macOS version:", err)
	} else if major < minMacOSMajor {
		report["status"] = "unsupported-platform"
		report["reason"] = fmt.Sprintf("macOS %d is below Sonoma 14.0", major)
		fmt.Fprintf(os.Stderr, "[FAIL] macOS %d is too old. Quenchforge requires Sonoma 14.0+.\n", major)
		exitCode = 2
		return
	}

	// Phase 3: hardware
	info, err := hardware.Detect()
	if err != nil {
		report["status"] = "probe-error"
		report["reason"] = err.Error()
		fmt.Fprintln(os.Stderr, "[FAIL] Hardware probe error:", err)
		exitCode = 1
		return
	}
	report["profile"] = string(info.Profile)
	report["gpu"] = info.GPU
	report["vram_gb"] = strconv.Itoa(info.GPUVRAMGB)
	report["metal"] = strconv.FormatBool(info.HasMetal)

	switch info.Profile {
	case hardware.ProfileVegaPro, hardware.ProfileW6800X, hardware.ProfileRDNA1,
		hardware.ProfileRDNA2, hardware.ProfileAppleSilicon:
		// supported
	case hardware.ProfileIGPU:
		// Works, but performance is awful. Warn and continue.
		fmt.Fprintln(os.Stderr, "[WARN] Intel Mac integrated GPU detected.")
		fmt.Fprintln(os.Stderr, "       Quenchforge will run but performance will be poor.")
		fmt.Fprintln(os.Stderr, "       A discrete AMD GPU is the recommended config.")
	case hardware.ProfileCPU, hardware.ProfileUnknown:
		report["status"] = "unsupported-hardware"
		report["reason"] = "no Metal device detected"
		fmt.Fprintln(os.Stderr, "[FAIL] No Metal-capable GPU detected.")
		fmt.Fprintln(os.Stderr, "       Quenchforge needs a discrete AMD GPU or Apple Silicon.")
		exitCode = 3
		return
	default:
		// Unknown but classify_profile says we have metal — proceed but warn.
		fmt.Fprintln(os.Stderr, "[WARN] Unrecognized profile:", info.Profile)
	}

	// Phase 4: disk
	freeGB, err := freeDiskGB("/")
	if err == nil {
		report["free_disk_gb"] = strconv.Itoa(freeGB)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "[WARN] Could not check free disk:", err)
	} else if freeGB < minFreeDiskGB {
		report["status"] = "insufficient-disk"
		report["reason"] = fmt.Sprintf("%d GB free; need %d GB", freeGB, minFreeDiskGB)
		fmt.Fprintf(os.Stderr, "[FAIL] %d GB free on /; quenchforge needs at least %d GB.\n",
			freeGB, minFreeDiskGB)
		exitCode = 4
		return
	}

	// All good
	report["status"] = "ok"
	fmt.Fprintln(os.Stderr, "[OK]   Quenchforge is supported on this system.")
	fmt.Fprintf(os.Stderr, "       profile=%s gpu=%q vram=%dGB free_disk=%dGB\n",
		info.Profile, info.GPU, info.GPUVRAMGB, freeGB)
}

// macOSMajor parses /System/Library/CoreServices/SystemVersion.plist to find
// the macOS major number. We avoid a plist parser dependency with a tiny
// substring scan — the plist format is stable across every macOS version
// we care about.
func macOSMajor() (int, error) {
	out, err := readFile("/System/Library/CoreServices/SystemVersion.plist")
	if err != nil {
		return 0, fmt.Errorf("read SystemVersion.plist: %w", err)
	}
	if v := extractPlistVersion(out); v > 0 {
		return v, nil
	}
	return 0, fmt.Errorf("could not parse macOS major from SystemVersion.plist")
}

// extractPlistVersion pulls "<key>ProductVersion</key><string>14.5</string>"
// out of /System/Library/CoreServices/SystemVersion.plist. We use a tiny
// substring scan rather than pulling in a plist parser dependency.
func extractPlistVersion(b []byte) int {
	s := string(b)
	const k = "<key>ProductVersion</key>"
	i := strings.Index(s, k)
	if i < 0 {
		return 0
	}
	rest := s[i+len(k):]
	start := strings.Index(rest, "<string>")
	if start < 0 {
		return 0
	}
	rest = rest[start+len("<string>"):]
	end := strings.Index(rest, "</string>")
	if end < 0 {
		return 0
	}
	v := rest[:end]
	if dot := strings.IndexByte(v, '.'); dot > 0 {
		v = v[:dot]
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0
	}
	return n
}

// freeDiskGB returns the free space (in GB) on the volume backing path.
func freeDiskGB(path string) (int, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	free := uint64(st.Bavail) * uint64(st.Bsize)
	return int(free >> 30), nil
}

// readFile is a thin wrapper so tests can spoof the plist contents.
var readFile = func(path string) ([]byte, error) {
	return os.ReadFile(path)
}
