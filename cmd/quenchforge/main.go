// Command quenchforge is the user-facing CLI and daemon entrypoint.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/discovery"
	"github.com/cerid-ai/quenchforge/internal/gateway"
	"github.com/cerid-ai/quenchforge/internal/hardware"
	"github.com/cerid-ai/quenchforge/internal/placement"
	"github.com/cerid-ai/quenchforge/internal/portcheck"
	"github.com/cerid-ai/quenchforge/internal/pressure"
	"github.com/cerid-ai/quenchforge/internal/supervisor"
	"github.com/cerid-ai/quenchforge/internal/tuning"
)

// Version is injected at build time via:
//
//	go build -ldflags "-X main.Version=$(git describe --tags --always)"
//
// goreleaser handles this in CI. Local dev builds carry the zero value.
var (
	Version   = "0.5.0-dev"
	Commit    = "unknown"
	BuildDate = "unknown"
)

const rootUsage = `quenchforge — local inference for Mac + AMD discrete GPU

Usage:
    quenchforge <command> [arguments]

Commands:
    serve              Start the HTTP gateway (Ollama + OpenAI compatible).
    install            Drop the LaunchAgent plist into ~/Library/LaunchAgents/ (macOS).
    doctor             Print a hardware-and-environment report for bug triage.
    pull               Download a GGUF model from HuggingFace.
    list               List GGUFs cached locally.
    rm                 Remove a cached GGUF.
    migrate-from-ollama  Symlink ~/.ollama/models/ GGUFs into the quenchforge cache.
    version            Print version, commit, and build date.
    help               Show this message.

Examples:
    quenchforge pull llama3.2:3b                                  # catalog alias
    quenchforge pull bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M  # explicit repo:quant
    quenchforge pull --list                                       # show catalog

Documentation: https://github.com/cerid-ai/quenchforge
Report a bug:  https://github.com/cerid-ai/quenchforge/issues/new/choose
`

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "quenchforge:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprint(stdout, rootUsage)
		return nil
	}

	switch args[0] {
	case "version", "--version", "-v":
		return cmdVersion(stdout)
	case "doctor":
		return cmdDoctor(args[1:], stdout, stderr)
	case "serve":
		return cmdServe(args[1:], stdout, stderr)
	case "install":
		return cmdInstall(args[1:], stdout, stderr)
	case "migrate-from-ollama":
		return cmdMigrate(args[1:], stdout, stderr)
	case "pull":
		return cmdPull(args[1:], stdout, stderr)
	case "list", "ls":
		return cmdList(args[1:], stdout, stderr)
	case "rm", "remove":
		return cmdRm(args[1:], stdout, stderr)
	case "help", "--help", "-h":
		fmt.Fprint(stdout, rootUsage)
		return nil
	default:
		fmt.Fprintf(stderr, "quenchforge: unknown command %q\n\n", args[0])
		fmt.Fprint(stderr, rootUsage)
		return fmt.Errorf("unknown command: %s", args[0])
	}
}

func cmdVersion(out io.Writer) error {
	fmt.Fprintf(out, "quenchforge %s\n", Version)
	fmt.Fprintf(out, "  commit:     %s\n", Commit)
	fmt.Fprintf(out, "  build date: %s\n", BuildDate)
	fmt.Fprintf(out, "  go:         %s\n", runtime.Version())
	fmt.Fprintf(out, "  platform:   %s/%s\n", runtime.GOOS, runtime.GOARCH)
	return nil
}

// ---------------------------------------------------------------------------
// doctor — real hardware probe + config + environment report.
// ---------------------------------------------------------------------------

func cmdDoctor(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	redacted := fs.Bool("redacted", false, "produce a paste-safe report (no usernames, no paths beyond ~)")
	explain := fs.Bool("explain", false, "append a Remediation block with per-finding action steps")
	if err := fs.Parse(args); err != nil {
		return err
	}

	fmt.Fprintln(stdout, "quenchforge doctor")
	fmt.Fprintln(stdout, "==================")
	fmt.Fprintf(stdout, "  version:    %s (%s)\n", Version, Commit)
	fmt.Fprintf(stdout, "  build date: %s\n", BuildDate)
	fmt.Fprintf(stdout, "  runtime:    go %s on %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(stdout, "  generated:  %s\n", time.Now().UTC().Format(time.RFC3339))
	fmt.Fprintln(stdout)

	if runtime.GOOS != "darwin" {
		fmt.Fprintln(stdout, "  status: UNSUPPORTED (quenchforge is macOS-only)")
		return nil
	}

	// Hardware probe
	info, hwErr := hardware.Detect()
	fmt.Fprintln(stdout, "hardware:")
	if hwErr != nil {
		fmt.Fprintf(stdout, "  PROBE FAILED: %v\n", hwErr)
	}
	fmt.Fprintf(stdout, "  profile:    %s\n", info.Profile)
	fmt.Fprintf(stdout, "  os:         %s\n", info.OSVersion)
	fmt.Fprintf(stdout, "  cpu:        %s\n", info.CPU)
	fmt.Fprintf(stdout, "  cpu cores:  %d\n", info.CPUCores)
	fmt.Fprintf(stdout, "  ram:        %d GB\n", info.TotalRAMGB)
	fmt.Fprintf(stdout, "  gpu:        %s\n", info.GPU)
	fmt.Fprintf(stdout, "  vram:       %d GB\n", info.GPUVRAMGB)
	fmt.Fprintf(stdout, "  metal:      %v\n", info.HasMetal)
	for i, d := range info.Devices {
		fmt.Fprintf(stdout, "    device[%d]: %q vram=%dGB lowpower=%v apple=%v\n",
			i, d.Name, d.VRAMGB, d.LowPower, d.AppleSilicon)
	}
	fmt.Fprintln(stdout)

	// Config
	cfg, cfgErr := config.Load()
	fmt.Fprintln(stdout, "config:")
	if cfgErr != nil {
		fmt.Fprintf(stdout, "  LOAD FAILED: %v\n", cfgErr)
	}
	fmt.Fprintf(stdout, "  listen addr:   %s\n", cfg.ListenAddr)
	fmt.Fprintf(stdout, "  llama bin:     %s\n", redactPath(cfg.LlamaBin, *redacted))
	fmt.Fprintf(stdout, "  models dir:    %s\n", redactPath(cfg.ModelsDir, *redacted))
	fmt.Fprintf(stdout, "  log dir:       %s\n", redactPath(cfg.LogDir, *redacted))
	fmt.Fprintf(stdout, "  pid dir:       %s\n", redactPath(cfg.PIDDir, *redacted))
	fmt.Fprintf(stdout, "  default model: %s\n", cfg.DefaultModel)
	fmt.Fprintf(stdout, "  max context:   %d\n", cfg.MaxContext)
	fmt.Fprintf(stdout, "  metal n_cb:    %d\n", cfg.MetalNCB)
	fmt.Fprintf(stdout, "  telemetry:     %v\n", cfg.TelemetryEnabled)
	fmt.Fprintln(stdout)

	// Slot config — operators read this to verify their env vars took
	// effect and to see at a glance which slots are configured to start.
	// Order mirrors gateway.SlotKind. Each "(opt-in)" annotation reflects
	// whether the slot starts unconditionally (chat) or only when its
	// model env var is set.
	fmt.Fprintln(stdout, "slots:")
	fmt.Fprintf(stdout, "  chat:         model=%s port=%d\n", cfg.DefaultModel, cfg.ChatPort)
	fmt.Fprintf(stdout, "  embed:        %s\n", slotLine(cfg.EmbedModel, cfg.EmbedPort))
	fmt.Fprintf(stdout, "  code-embed:   %s   (routed by request model == cfg.CodeEmbedModel)\n",
		slotLine(cfg.CodeEmbedModel, cfg.CodeEmbedPort))
	fmt.Fprintf(stdout, "  rerank:       %s\n", slotLine(cfg.RerankModel, cfg.RerankPort))
	fmt.Fprintf(stdout, "  whisper:      %s\n", slotLine(cfg.WhisperModel, cfg.WhisperPort))
	fmt.Fprintf(stdout, "  imagegen (sd):%s\n", slotLine(cfg.SDModel, cfg.SDPort))
	fmt.Fprintf(stdout, "  tts (bark):   %s\n", slotLine(cfg.BarkModel, cfg.BarkPort))
	fmt.Fprintln(stdout)

	// llama-server binary check
	llamaBin, llamaErr := resolveLlamaBin(cfg.LlamaBin)
	fmt.Fprintln(stdout, "binaries:")
	if llamaErr != nil {
		fmt.Fprintf(stdout, "  llama-server: NOT FOUND (%v)\n", llamaErr)
		fmt.Fprintln(stdout, "    hint: run scripts/build-llama.sh or set QUENCHFORGE_LLAMA_BIN")
	} else {
		fmt.Fprintf(stdout, "  llama-server: %s\n", redactPath(llamaBin, *redacted))
	}
	fmt.Fprintln(stdout)

	// Model registry summary
	models, _ := gateway.EnumerateModels(cfg.ModelsDir)
	fmt.Fprintln(stdout, "models:")
	if len(models) == 0 {
		fmt.Fprintln(stdout, "  (no GGUFs cached)")
		fmt.Fprintln(stdout, "  hint: try `quenchforge migrate-from-ollama` if you have an Ollama install")
	} else {
		fmt.Fprintf(stdout, "  count: %d\n", len(models))
		for _, m := range models {
			fmt.Fprintf(stdout, "    %s (%.2f GB)\n", m.Name, float64(m.SizeBytes)/(1<<30))
		}
	}
	fmt.Fprintln(stdout)

	// ------------------------------------------------------------------
	// v0.7.2 Layer 1b additions. Each new section is additive (appended
	// after the existing report) so existing bug-report parsers continue
	// to work. Renaming any of the four section headers below is a
	// breaking change for the .github/ISSUE_TEMPLATE/bug.yml triage
	// contract (see CLAUDE.md absolute rule 4).
	// ------------------------------------------------------------------

	// Ollama LaunchAgent — surfaces the single most common public-install
	// conflict: the com.ollama.ollama login agent racing quenchforge for
	// port 11434 at every login.
	fmt.Fprintln(stdout, "Ollama LaunchAgent")
	fmt.Fprintln(stdout, "------------------")
	fmt.Fprintf(stdout, "  status: %s\n", checkOllamaLaunchAgent())
	fmt.Fprintln(stdout)

	// Disk space — the 2026-05-24 freeze cascaded out of a disk-full
	// state that prevented APFS from writing kernel panic reports. WARN
	// at 20 GB, CRITICAL at 10 GB, on /System/Volumes/Data (the actual
	// writable APFS volume on macOS, distinct from the read-only system
	// volume).
	fmt.Fprintln(stdout, "Disk space")
	fmt.Fprintln(stdout, "----------")
	fmt.Fprintln(stdout, "  "+checkDiskFree("/System/Volumes/Data"))
	fmt.Fprintln(stdout)

	// Slot log sizes — Layer 1c rotation should keep these bounded, but
	// pre-rotation installs (or installs that set QUENCHFORGE_LOG_MAX_BYTES=0)
	// can still drift unbounded. Sorted by size desc so the worst offender
	// is the first line operators see.
	fmt.Fprintln(stdout, "Slot log sizes")
	fmt.Fprintln(stdout, "--------------")
	for _, line := range checkSlotLogSizes() {
		fmt.Fprintln(stdout, "  "+line)
	}
	fmt.Fprintln(stdout)

	// Port 11434 — uses the Phase 2 portcheck package. The classification
	// is identical to what cmdServe uses pre-bind, so doctor output and
	// the runtime behavior stay in sync.
	fmt.Fprintln(stdout, "Port 11434")
	fmt.Fprintln(stdout, "----------")
	{
		pcCtx, pcCancel := context.WithTimeout(context.Background(), 3*time.Second)
		res, err := portcheck.Check(pcCtx, "127.0.0.1:11434")
		pcCancel()
		if err != nil {
			fmt.Fprintf(stdout, "  could not probe: %v\n", err)
		} else {
			switch res.Verdict {
			case portcheck.VerdictFree:
				fmt.Fprintln(stdout, "  free — quenchforge will bind on next start")
			case portcheck.VerdictHeldByOllama:
				fmt.Fprintf(stdout, "  held by Ollama (pid %d, %s) — CRITICAL\n",
					res.Holder.PID, res.Holder.ExecPath)
			case portcheck.VerdictHeldByStaleQuenchforge:
				fmt.Fprintf(stdout, "  held by quenchforge (pid %d) — OK\n", res.Holder.PID)
			case portcheck.VerdictHeldByOther:
				fmt.Fprintf(stdout, "  held by pid %d (%s, %s) — WARN\n",
					res.Holder.PID, res.Holder.CommandName, res.Holder.ExecPath)
			case portcheck.VerdictUnknown:
				fmt.Fprintln(stdout, "  in use but holder could not be identified — WARN")
			}
		}
	}
	fmt.Fprintln(stdout)

	// --explain — per-finding remediation steps, so a bug-report triage
	// reply can quote the doctor output AND the next action in one paste.
	// Framed as a "Common remediations" reference (not "Remediation") so
	// a user with a healthy system reading the section understands these
	// are conditional steps, not actions they should take unconditionally.
	// Gating per-line on each helper's status would require threading
	// structured status through every check — a larger refactor — and
	// this framing keeps the diff small while removing the misleading
	// "fix this" framing.
	if *explain {
		fmt.Fprintln(stdout, "Common remediations")
		fmt.Fprintln(stdout, "-------------------")
		fmt.Fprintln(stdout, "  (Reference — apply only those matching non-PASS findings above.)")
		fmt.Fprintln(stdout, `  - Ollama LaunchAgent loaded:    launchctl bootout gui/$(id -u)/com.ollama.ollama`)
		fmt.Fprintln(stdout, `  - Disk space CRITICAL:          docker system prune -a --volumes && truncate slot logs`)
		fmt.Fprintln(stdout, `  - Slot log CRITICAL:            : > <path-to-log>  (Layer 1c rotation should prevent recurrence)`)
		fmt.Fprintln(stdout, `  - Port 11434 held by Ollama:    see "Ollama LaunchAgent" remediation`)
		fmt.Fprintln(stdout, `  - Port 11434 unknown holder:    lsof -i :11434 -sTCP:LISTEN  (resolve manually)`)
		fmt.Fprintln(stdout)
	}

	fmt.Fprintln(stdout, "Paste this output verbatim into bug reports.")
	return nil
}

// checkOllamaLaunchAgent reports whether com.ollama.ollama is loaded
// in the user's GUI launchd domain. Returns one of:
//   - "not installed"      (Ollama.app not present)
//   - "disabled"           (plist present but launchctl bootout has been
//     applied, OR app present but agent never loaded)
//   - "loaded (PID N) — DISARM with: ..." (currently running)
//   - "loaded but stopped" (loaded, no PID — exited cleanly)
//
// Implementation uses `launchctl list com.ollama.ollama`: exit 0 with
// output if loaded, exit nonzero if not. The PID lives in launchctl's
// list-output as the `"PID" = N;` line.
func checkOllamaLaunchAgent() string {
	// Is Ollama.app even installed?
	if _, err := os.Stat("/Applications/Ollama.app"); os.IsNotExist(err) {
		return "not installed"
	}

	// Bounded so a hung launchd doesn't make `quenchforge doctor` itself
	// hang — diagnostic tools must always return quickly.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "launchctl", "list", "com.ollama.ollama").Output()
	if err != nil {
		return "disabled"
	}
	for _, ln := range strings.Split(string(out), "\n") {
		ln = strings.TrimSpace(ln)
		// Apple's launchctl list output has used both `"PID" = N;` (quoted)
		// and `PID = N;` (unquoted) depending on macOS version + launchd
		// domain. Tolerate either. Trailing space on the unquoted form
		// avoids matching unrelated keys like `PIDProcess`.
		if !strings.HasPrefix(ln, `"PID"`) && !strings.HasPrefix(ln, "PID ") {
			continue
		}
		// "PID" = 1253;
		parts := strings.SplitN(ln, "=", 2)
		if len(parts) != 2 {
			continue
		}
		v := strings.TrimSpace(parts[1])
		v = strings.TrimSuffix(v, ";")
		if pid, err := strconv.Atoi(v); err == nil && pid > 0 {
			return fmt.Sprintf("loaded (PID %d) — DISARM with: launchctl bootout gui/$(id -u)/com.ollama.ollama", pid)
		}
	}
	return "loaded but stopped"
}

// checkDiskFree returns a human-readable line for the given mount
// point with a PASS/WARN/CRITICAL hint. Uses `df -k` so we avoid
// syscall.Statfs CGo complications across macOS versions. WARN at
// < 20 GB, CRITICAL at < 10 GB — the thresholds that map to "kernel
// panic reports can no longer be written" in the 2026-05-24 RCA.
func checkDiskFree(mount string) string {
	// Bounded so a hung volume mount (e.g. an unresponsive network share
	// in the filesystem table) doesn't make `quenchforge doctor` itself
	// hang — diagnostic tools must always return quickly.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "df", "-k", mount).Output()
	if err != nil {
		return fmt.Sprintf("could not read disk usage for %s: %v", mount, err)
	}
	// df -k output:
	//   Filesystem  1024-blocks  Used  Available  Capacity  Mounted on
	//   /dev/disk4s2  847249000  799000000  48000000  95%  /System/Volumes/Data
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) < 2 {
		return "df returned unexpected output"
	}
	fields := strings.Fields(lines[1])
	if len(fields) < 5 {
		return "df row has fewer than 5 fields"
	}
	availKB, err := strconv.ParseInt(fields[3], 10, 64)
	if err != nil {
		return fmt.Sprintf("could not parse df Available column: %q", fields[3])
	}
	availGB := availKB / 1024 / 1024
	capacity := fields[4]
	status := "PASS"
	switch {
	case availGB < 10:
		status = "CRITICAL"
	case availGB < 20:
		status = "WARN"
	}
	return fmt.Sprintf("%s: %d GB available (%s capacity) — %s", mount, availGB, capacity, status)
}

// checkSlotLogSizes returns one line per file under
// ~/Library/Logs/quenchforge/, sorted by size desc. Each line carries
// a PASS/WARN/CRITICAL marker:
//   - PASS:     < 50 MB
//   - WARN:     50–500 MB
//   - CRITICAL: > 500 MB
func checkSlotLogSizes() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return []string{fmt.Sprintf("could not resolve home dir: %v", err)}
	}
	dir := filepath.Join(home, "Library", "Logs", "quenchforge")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{"(no slot logs yet)"}
		}
		return []string{fmt.Sprintf("could not read %s: %v", dir, err)}
	}
	type entry struct {
		name string
		size int64
	}
	var es []entry
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		es = append(es, entry{e.Name(), info.Size()})
	}
	sort.Slice(es, func(i, j int) bool { return es[i].size > es[j].size })

	var out []string
	for _, e := range es {
		mb := e.size / 1024 / 1024
		status := "PASS"
		switch {
		case mb > 500:
			status = "CRITICAL"
		case mb > 50:
			status = "WARN"
		}
		out = append(out, fmt.Sprintf("%-40s %6d MB  %s", e.name, mb, status))
	}
	if len(out) == 0 {
		return []string{"(no slot logs yet)"}
	}
	return out
}

// ---------------------------------------------------------------------------
// serve — gateway + (optional) supervised llama-server slot.
// ---------------------------------------------------------------------------

func cmdServe(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	listen := fs.String("listen", "", "override listen address (env: QUENCHFORGE_LISTEN_ADDR)")
	noSlot := fs.Bool("no-slot", false, "start the gateway only; don't supervise any llama-server slots")
	model := fs.String("model", "", "GGUF model name (under models dir) to load in the chat slot; "+
		"empty = read from QUENCHFORGE_DEFAULT_MODEL")
	embedModel := fs.String("embed-model", "", "GGUF model name to load in the embedding slot; "+
		"empty = read from QUENCHFORGE_EMBED_MODEL (no embed slot when both are empty)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	if *listen != "" {
		cfg.ListenAddr = *listen
	}
	if *embedModel != "" {
		cfg.EmbedModel = *embedModel
	}
	if err := cfg.EnsureDirs(); err != nil {
		return err
	}

	// Pre-bind port check. Identifies common conflicts (Ollama, stale
	// quenchforge) and exits cleanly with actionable guidance so the
	// LaunchAgent's KeepAlive=<dict><SuccessfulExit false/></dict>
	// (post-Layer-2 plist) does not respawn-loop.
	{
		pcCtx, pcCancel := context.WithTimeout(context.Background(), 5*time.Second)
		res, err := portcheck.Check(pcCtx, cfg.ListenAddr)
		pcCancel()
		if err != nil {
			fmt.Fprintf(stderr, "quenchforge: port probe failed for %s: %v\n", cfg.ListenAddr, err)
			// Probe failure is not fatal — fall through to bind and let
			// the real error surface if there is one.
		} else {
			switch res.Verdict {
			case portcheck.VerdictFree:
				// No-op.
			case portcheck.VerdictHeldByOllama:
				fmt.Fprint(stderr, portcheck.FormatOllamaMessage(cfg.ListenAddr, res.Holder.PID))
				return nil // clean exit so launchd does not respawn
			case portcheck.VerdictHeldByStaleQuenchforge:
				fmt.Fprintf(stderr,
					"quenchforge: stale quenchforge process (pid %d) is holding %s. "+
						"Try: kill %d && launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge\n",
					res.Holder.PID, cfg.ListenAddr, res.Holder.PID)
				return nil
			case portcheck.VerdictHeldByOther:
				fmt.Fprintf(stderr,
					"quenchforge: port %s is held by pid %d (%s, %s). "+
						"Resolve the conflict and re-run.\n",
					cfg.ListenAddr, res.Holder.PID, res.Holder.CommandName, res.Holder.ExecPath)
				return nil
			case portcheck.VerdictUnknown:
				// Build a valid `lsof -i :PORT` hint — cfg.ListenAddr is
				// "host:port" and lsof's -i syntax wants the port alone.
				// Fall back to the raw addr if SplitHostPort fails.
				lsofTarget := cfg.ListenAddr
				if _, port, err := net.SplitHostPort(cfg.ListenAddr); err == nil {
					lsofTarget = port
				}
				fmt.Fprintf(stderr,
					"quenchforge: port %s is in use but holder could not be identified "+
						"(lsof + netstat both failed). Check `lsof -i :%s` manually.\n",
					cfg.ListenAddr, lsofTarget)
				return nil
			}
		}
	}

	// Resolve per-slot log rotation parameters ONCE per cmdServe; we
	// thread the values through every slot constructor so operators
	// only pay one env-var lookup per process start. Previously each
	// startSlot / inline imagegen|tts|whisper block re-parsed the env,
	// and any drift in default values between callers would diverge
	// silently.
	maxLogBytes, logBackups := slotLogRotation()

	// Hardware probe — gates per-profile slot tuning. Best-effort: a probe
	// failure falls back to the unknown profile, which uses the default
	// (Apple-Silicon-friendly) llama-server args. The slot is still
	// supervised so degraded inference is better than no inference.
	hwInfo, hwErr := hardware.Detect()
	if hwErr != nil {
		fmt.Fprintf(stderr,
			"quenchforge: warning: hardware probe failed: %v\n"+
				"  Slot tuning defaults to apple-silicon profile; chat may be\n"+
				"  unstable on AMD discrete. Run `quenchforge doctor` to diagnose.\n",
			hwErr)
	}
	if hwInfo.IsAMDDiscrete() {
		fmt.Fprintf(stdout,
			"quenchforge: detected %s profile — chat slot will use "+
				"--flash-attn off --cache-ram 0 --no-cache-prompt\n"+
				"  (avoids GPU↔CPU flash-attn fallback throttling + Vega II "+
				"prompt-cache GGML_ASSERT crash; embed/rerank slots are unaffected)\n",
			hwInfo.Profile)
	}

	// VRAM pre-flight (v0.4.0). Refuse to spawn slots whose combined
	// model weights would over-subscribe VRAM. Operator-friendly error
	// is better than three Metal-load failures in a row.
	if !*noSlot && !vramCheckDisabled() {
		if err := checkVRAMBudget(cfg, hwInfo, stdout); err != nil {
			fmt.Fprintf(stderr, "quenchforge: %v\n", err)
			return fmt.Errorf("VRAM pre-flight failed")
		}
	}

	// Orphan reaper — clean up any survivors from a previous crash before we
	// allocate new ports.
	reaped := supervisor.ReapOrphans(cfg.PIDDir)
	for _, r := range reaped {
		fmt.Fprintf(stderr, "quenchforge: reap %s pid=%d action=%s %s\n",
			r.File, r.PID, r.Action, r.Note)
	}

	ctx, cancel := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	g := gateway.New(cfg)
	g.SetVersion(Version)
	// Install the device-placement policy so the gateway can route "auto"
	// embedding kinds per request and skip GPU admission for CPU-placed kinds.
	// Built from the same hardware profile + operator overrides the tuning
	// module uses, so placement and slot tuning never disagree.
	pol := tuning.PolicyFor(hwInfo.Profile, cfg)
	g.SetPlacement(pol, cfg.AutoBatchThreshold)
	if cfg.GovernorEnabled {
		g.SetScheduler(startGovernor(ctx, pressure.NewSensor(), cfg, stdout))
	}
	if err := g.Start(ctx); err != nil {
		if errors.Is(err, gateway.ErrAddrInUse) {
			fmt.Fprintf(stderr,
				"quenchforge: port %s is already in use.\n"+
					"  Is Ollama running? Try: `brew services stop ollama`\n"+
					"  Or pick a different port: `quenchforge serve --listen 127.0.0.1:11435`\n",
				cfg.ListenAddr)
		}
		return err
	}
	fmt.Fprintf(stdout, "quenchforge: gateway listening on http://%s\n", cfg.ListenAddr)

	// mDNS advertisement (opt-in via QUENCHFORGE_ADVERTISE_MDNS). The
	// gateway's listen port is what gets advertised — so peers see the
	// HTTP surface, not the raw llama-server slots.
	var mdnsAdv discovery.Advertiser
	if cfg.AdvertiseMDNS {
		port, parseErr := portFromListenAddr(cfg.ListenAddr)
		if parseErr != nil {
			fmt.Fprintf(stderr, "quenchforge: warning: mDNS skipped: %v\n", parseErr)
		} else {
			adv, err := discovery.Start(ctx, discovery.Service{
				Port: port,
				TXTRecords: []string{
					"version=" + Version,
					"api=ollama,openai",
				},
			})
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: mDNS advertisement failed: %v\n"+
						"  Gateway still works; peers just won't auto-discover it.\n", err)
			} else {
				mdnsAdv = adv
				fmt.Fprintln(stdout,
					"quenchforge: advertised on Bonjour as `_quenchforge._tcp.local.`")
				fmt.Fprintln(stdout,
					"  (Sonoma+: first launch may show a 'find devices on your local network' prompt)")
			}
		}
	}

	// Slots are tracked in a small map so the shutdown path handles them
	// uniformly. The map key matches the gateway SlotKind so logs and
	// the `quenchforge doctor` report stay readable.
	//
	// The `--no-slot` flag suppresses the chat slot only. Embed / rerank /
	// whisper are opt-in via their model env vars, so they evaluate
	// independently — handy for headless transcription deployments that
	// don't want a chat slot eating VRAM.
	slots := map[gateway.SlotKind]*supervisor.Slot{}

	// startEmbedFamily launches the embed/code-embed slot(s) for one kind.
	// Under "auto" placement it brings up TWO instances — a GPU instance
	// (batched throughput, registered as the primary upstream) and a CPU
	// instance (single-request latency, registered as the CPU upstream) — so
	// the gateway routes per request by batch shape. For fixed gpu/cpu
	// placement it launches a single instance and lets the policy pick the
	// device, identical to the pre-auto single-slot path.
	startEmbedFamily := func(kind gateway.SlotKind, name, model string, gpuPort, cpuPort int) {
		extra := []string{"--embedding", "--pooling", "cls"}
		if pol.Mode(string(kind)) != placement.ModeAuto {
			s, err := startSlot(ctx, cfg, hwInfo, slotSpec{
				Kind: kind, Name: name, Model: model, Port: gpuPort, ExtraArgs: extra,
			}, maxLogBytes, logBackups, stderr)
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: %s slot not started: %v\n", name, err)
				return
			}
			slots[kind] = s
			fmt.Fprintf(stdout, "quenchforge: %s slot pid=%d model=%s port=%d\n",
				name, s.PID(), model, gpuPort)
			_ = g.SetUpstream(kind, fmt.Sprintf("http://127.0.0.1:%d", gpuPort))
			return
		}
		// Auto: dual-placed. GPU instance is the primary upstream.
		gpuDev, cpuDev := placement.GPU, placement.CPU
		if s, err := startSlot(ctx, cfg, hwInfo, slotSpec{
			Kind: kind, Name: name, Model: model, Port: gpuPort, ExtraArgs: extra, Device: &gpuDev,
		}, maxLogBytes, logBackups, stderr); err != nil {
			fmt.Fprintf(stderr,
				"quenchforge: warning: %s (gpu) slot not started: %v\n", name, err)
		} else {
			slots[kind] = s
			fmt.Fprintf(stdout, "quenchforge: %s slot (gpu) pid=%d model=%s port=%d\n",
				name, s.PID(), model, gpuPort)
			_ = g.SetUpstream(kind, fmt.Sprintf("http://127.0.0.1:%d", gpuPort))
		}
		// CPU instance handles single-request latency traffic.
		cpuName := name + "-cpu"
		if s, err := startSlot(ctx, cfg, hwInfo, slotSpec{
			Kind: kind, Name: cpuName, Model: model, Port: cpuPort, ExtraArgs: extra, Device: &cpuDev,
		}, maxLogBytes, logBackups, stderr); err != nil {
			fmt.Fprintf(stderr,
				"quenchforge: warning: %s slot not started: %v\n"+
					"  Single-input %s requests will route to the GPU instance.\n",
				cpuName, err, name)
		} else {
			slots[gateway.SlotKind(cpuName)] = s
			fmt.Fprintf(stdout, "quenchforge: %s slot (cpu) pid=%d model=%s port=%d\n",
				cpuName, s.PID(), model, cpuPort)
			_ = g.SetCPUUpstream(kind, fmt.Sprintf("http://127.0.0.1:%d", cpuPort))
		}
	}

	if !*noSlot {
		// Chat slot — always-on unless suppressed.
		modelName := *model
		if modelName == "" {
			modelName = cfg.DefaultModel
		}
		s, err := startSlot(ctx, cfg, hwInfo, slotSpec{
			Kind:      gateway.KindChat,
			Name:      "chat",
			Model:     modelName,
			Port:      cfg.ChatPort,
			ExtraArgs: nil,
		}, maxLogBytes, logBackups, stderr)
		if err != nil {
			fmt.Fprintf(stderr,
				"quenchforge: warning: chat slot not started: %v\n"+
					"  Gateway is up but /api/chat will return 503.\n"+
					"  Run `quenchforge doctor` for the binary path and model registry.\n", err)
		} else {
			slots[gateway.KindChat] = s
			fmt.Fprintf(stdout, "quenchforge: chat slot pid=%d model=%s port=%d\n",
				s.PID(), modelName, cfg.ChatPort)
			_ = g.SetUpstream(gateway.KindChat,
				fmt.Sprintf("http://127.0.0.1:%d", cfg.ChatPort))
		}
	}
	{ // embed/rerank/whisper evaluate independently of --no-slot

		// Embed slot — opt-in via QUENCHFORGE_EMBED_MODEL or --embed-model.
		// --embedding flips llama-server into pooled-embedding mode and
		// --pooling cls is the standard for most BERT-style embedders (added
		// by startEmbedFamily). Under "auto" placement this brings up a
		// GPU+CPU pair; otherwise a single policy-placed instance.
		if cfg.EmbedModel != "" {
			startEmbedFamily(gateway.KindEmbed, "embed", cfg.EmbedModel,
				cfg.EmbedPort, cfg.EmbedCPUPort)
		}

		// Code-embed slot — opt-in via QUENCHFORGE_CODE_EMBED_MODEL.
		// Sibling to the regular embed slot; the gateway dispatches
		// /api/embeddings and /v1/embeddings requests here when the body's
		// `model` field equals cfg.CodeEmbedModel. Lets one quenchforge
		// process serve a general-text embedder (for KB / RAG) alongside
		// a code-tuned embedder (for semantic-code-search MCPs).
		if cfg.CodeEmbedModel != "" {
			startEmbedFamily(gateway.KindCodeEmbed, "code-embed", cfg.CodeEmbedModel,
				cfg.CodeEmbedPort, cfg.CodeEmbedCPUPort)
		}

		// Rerank slot — opt-in via QUENCHFORGE_RERANK_MODEL. Same
		// llama-server binary as chat/embed, just --reranking mode.
		if cfg.RerankModel != "" {
			s, err := startSlot(ctx, cfg, hwInfo, slotSpec{
				Kind:      gateway.KindRerank,
				Name:      "rerank",
				Model:     cfg.RerankModel,
				Port:      cfg.RerankPort,
				ExtraArgs: []string{"--reranking"},
			}, maxLogBytes, logBackups, stderr)
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: rerank slot not started: %v\n"+
						"  /v1/rerank will return 503.\n", err)
			} else {
				slots[gateway.KindRerank] = s
				fmt.Fprintf(stdout, "quenchforge: rerank slot pid=%d model=%s port=%d\n",
					s.PID(), cfg.RerankModel, cfg.RerankPort)
				_ = g.SetUpstream(gateway.KindRerank,
					fmt.Sprintf("http://127.0.0.1:%d", cfg.RerankPort))
			}
		}

		// Image-gen slot — opt-in via QUENCHFORGE_SD_MODEL. Uses sd-server
		// from stable-diffusion.cpp/examples/server/. Speaks OpenAI's
		// /v1/images/generations natively, so the gateway just proxies.
		if cfg.SDModel != "" {
			sdBin, err := resolveSDBin()
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: image-gen slot not started: %v\n"+
						"  /v1/images/generations will return 503.\n", err)
			} else {
				slot := supervisor.NewSlot("imagegen")
				slot.BinPath = sdBin
				slot.LogDir = cfg.LogDir
				slot.PIDDir = cfg.PIDDir
				slot.Args = []string{
					"-m", cfg.SDModel,
					"--listen-ip", "127.0.0.1",
					"--listen-port", fmt.Sprintf("%d", cfg.SDPort),
				}
				slot.Env = []string{fmt.Sprintf("GGML_METAL_N_CB=%d", cfg.MetalNCB)}
				slot.MaxLogBytes, slot.LogBackups = maxLogBytes, logBackups
				if err := slot.Start(ctx); err != nil {
					fmt.Fprintf(stderr,
						"quenchforge: warning: image-gen slot start failed: %v\n", err)
				} else {
					slots[gateway.KindImageGen] = slot
					fmt.Fprintf(stdout,
						"quenchforge: image-gen slot pid=%d model=%s port=%d\n",
						slot.PID(), cfg.SDModel, cfg.SDPort)
					_ = g.SetUpstream(gateway.KindImageGen,
						fmt.Sprintf("http://127.0.0.1:%d", cfg.SDPort))
				}
			}
		}

		// TTS slot — opt-in via QUENCHFORGE_BARK_MODEL. Uses the bark.cpp
		// server example. Its native route is /tts; the gateway rewrites
		// /v1/audio/speech to /tts on the way through.
		if cfg.BarkModel != "" {
			barkBin, err := resolveBarkBin()
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: TTS slot not started: %v\n"+
						"  /v1/audio/speech will return 503.\n", err)
			} else {
				slot := supervisor.NewSlot("tts")
				slot.BinPath = barkBin
				slot.LogDir = cfg.LogDir
				slot.PIDDir = cfg.PIDDir
				slot.Args = []string{
					"-m", cfg.BarkModel,
					"-a", "127.0.0.1",
					"-p", fmt.Sprintf("%d", cfg.BarkPort),
					"-t", "8",
				}
				slot.Env = []string{fmt.Sprintf("GGML_METAL_N_CB=%d", cfg.MetalNCB)}
				slot.MaxLogBytes, slot.LogBackups = maxLogBytes, logBackups
				if err := slot.Start(ctx); err != nil {
					fmt.Fprintf(stderr,
						"quenchforge: warning: TTS slot start failed: %v\n", err)
				} else {
					slots[gateway.KindTTS] = slot
					fmt.Fprintf(stdout,
						"quenchforge: TTS slot pid=%d model=%s port=%d\n",
						slot.PID(), cfg.BarkModel, cfg.BarkPort)
					_ = g.SetUpstream(gateway.KindTTS,
						fmt.Sprintf("http://127.0.0.1:%d", cfg.BarkPort))
				}
			}
		}

		// Whisper slot — opt-in via QUENCHFORGE_WHISPER_MODEL. Uses
		// whisper-server (a separate binary built from whisper.cpp).
		// Default to CPU on Mac because whisper's Metal path still has
		// AMD-specific bugs even with our patch applied — operators
		// can flip QUENCHFORGE_WHISPER_GPU=true to opt in if they're
		// on a hardware path where it works.
		if cfg.WhisperModel != "" {
			whisperBin, err := resolveWhisperBin()
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: whisper slot not started: %v\n"+
						"  /v1/audio/transcriptions will return 503.\n", err)
			} else {
				args := []string{
					"--model", cfg.WhisperModel,
					"--host", "127.0.0.1",
					"--port", fmt.Sprintf("%d", cfg.WhisperPort),
					"--threads", "8",
				}
				if !cfg.WhisperGPU {
					args = append(args, "--no-gpu")
				}
				slot := supervisor.NewSlot("whisper")
				slot.BinPath = whisperBin
				slot.LogDir = cfg.LogDir
				slot.PIDDir = cfg.PIDDir
				slot.Args = args
				slot.Env = []string{fmt.Sprintf("GGML_METAL_N_CB=%d", cfg.MetalNCB)}
				slot.MaxLogBytes, slot.LogBackups = maxLogBytes, logBackups
				if err := slot.Start(ctx); err != nil {
					fmt.Fprintf(stderr,
						"quenchforge: warning: whisper slot start failed: %v\n", err)
				} else {
					slots[gateway.KindWhisper] = slot
					fmt.Fprintf(stdout,
						"quenchforge: whisper slot pid=%d model=%s port=%d gpu=%v\n",
						slot.PID(), cfg.WhisperModel, cfg.WhisperPort, cfg.WhisperGPU)
					_ = g.SetUpstream(gateway.KindWhisper,
						fmt.Sprintf("http://127.0.0.1:%d", cfg.WhisperPort))
				}
			}
		}
	}

	// Wait for shutdown signal
	<-ctx.Done()
	fmt.Fprintln(stdout, "quenchforge: shutdown signal received")

	// Tear down: mDNS first (withdraws the advertisement before peers see
	// us go offline), then slots, then gateway. Grace timeouts on each so
	// a hung child can't block our exit indefinitely.
	if mdnsAdv != nil {
		_ = mdnsAdv.Stop()
	}
	for kind, slot := range slots {
		if err := slot.Stop(5 * time.Second); err != nil {
			fmt.Fprintf(stderr, "quenchforge: %s slot stop: %v\n", kind, err)
		}
	}
	if err := g.Stop(2 * time.Second); err != nil {
		fmt.Fprintf(stderr, "quenchforge: gateway stop: %v\n", err)
	}
	return nil
}

// portFromListenAddr pulls the port from a "host:port" listen address.
func portFromListenAddr(addr string) (int, error) {
	_, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return 0, fmt.Errorf("parse listen addr %q: %w", addr, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return 0, fmt.Errorf("parse port %q: %w", portStr, err)
	}
	return port, nil
}

// slotSpec describes one llama-server instance to launch.
type slotSpec struct {
	Kind      gateway.SlotKind
	Name      string // pidfile/log filename
	Model     string // model name under cfg.ModelsDir
	Port      int    // 127.0.0.1:Port
	ExtraArgs []string
	// Device, when non-nil, pins this instance to a specific compute device,
	// bypassing the placement policy. The supervisor sets it for the two
	// instances of an "auto"-placed kind (one GPU, one CPU). nil = unset = let
	// the policy decide (the default, behaviour-preserving path).
	Device *placement.Device
}

// specTuning returns the kernel tuning for a slot spec, honouring an explicit
// spec.Device when set (the dual-launch "auto" instances) and otherwise letting
// the placement policy decide (every existing single-slot caller).
func specTuning(cfg config.Config, hwInfo hardware.Info, spec slotSpec) tuning.SlotTuning {
	if spec.Device != nil {
		return tuning.KernelParamsForDevice(hwInfo.Profile, hwInfo.GPUVRAMGB, spec.Kind, cfg, *spec.Device)
	}
	return tuning.KernelParams(hwInfo.Profile, hwInfo.GPUVRAMGB, spec.Kind, cfg)
}

// buildSlotArgs constructs the llama-server command-line arguments for
// one supervised slot. Pure function (no I/O, no globals) so it can be
// unit-tested without spawning llama-server.
//
// The per-(profile, kind) tuning logic — AMD chat safety flags, embed
// batch overrides, rerank batch overrides, future per-profile knobs —
// lives in `internal/tuning/tuning.go`. This function is responsible
// only for the base arg shape plus the layering of the tuning result.
// Move the per-profile decisions there when they change, not here.
func buildSlotArgs(cfg config.Config, hwInfo hardware.Info, spec slotSpec, modelPath string) []string {
	tn := specTuning(cfg, hwInfo, spec)

	// VRAM-tier-adaptive context ceiling: ContextSize only ever lowers
	// cfg.MaxContext (small AMD cards), never raises it.
	ctxSize := cfg.MaxContext
	if tn.ContextSize > 0 && tn.ContextSize < ctxSize {
		ctxSize = tn.ContextSize
	}

	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", spec.Port),
		"--ctx-size", fmt.Sprintf("%d", ctxSize),
	}
	args = append(args, spec.ExtraArgs...)

	if tn.BatchSize > 0 {
		args = append(args, "--batch-size", fmt.Sprintf("%d", tn.BatchSize))
	}
	if tn.UbatchSize > 0 {
		args = append(args, "--ubatch-size", fmt.Sprintf("%d", tn.UbatchSize))
	}
	args = append(args, tn.ExtraArgs...)

	return args
}

// slotEnv assembles the per-slot environment variable list. Returns the
// list `Slot.Env` should be set to. Pure function for testability.
//
// GGML_METAL_N_CB is per-slot (tuning may override the global default
// for an embed/rerank slot to serialise Metal command-buffer submission
// on AMD discrete).
func slotEnv(cfg config.Config, hwInfo hardware.Info, kind gateway.SlotKind) []string {
	return envFromTuning(cfg, tuning.KernelParams(hwInfo.Profile, hwInfo.GPUVRAMGB, kind, cfg))
}

// envFromTuning derives the per-slot env list from a resolved SlotTuning. Split
// out of slotEnv so the dual-launch path (which resolves tuning per explicit
// device via specTuning) and the policy path produce identical env shapes.
func envFromTuning(cfg config.Config, tn tuning.SlotTuning) []string {
	ncb := cfg.MetalNCB
	if tn.MetalNCB > 0 {
		ncb = tn.MetalNCB
	}
	env := []string{
		fmt.Sprintf("GGML_METAL_N_CB=%d", ncb),
	}
	if tn.MetalConcurrencyDisable {
		env = append(env, "GGML_METAL_CONCURRENCY_DISABLE=1")
	}
	return env
}

// Per-slot log rotation defaults. An unattended embed.log on a Vega II
// install reached 3.73 GB in 7 days of uptime before bounded rotation
// landed; combined with Docker.raw growth that's how the dev machine
// reached 97% full and stopped being able to write kernel panic dumps.
// 100 MB × 5 backups caps each slot at ≈600 MB total on disk.
const (
	defaultSlotLogMaxBytes = 100 * 1024 * 1024 // 100 MB
	defaultSlotLogBackups  = 5
)

// slotLogRotation returns the rotation parameters every slot uses,
// resolved from QUENCHFORGE_LOG_MAX_BYTES / QUENCHFORGE_LOG_BACKUPS
// env vars with the defaults above. Setting QUENCHFORGE_LOG_MAX_BYTES=0
// disables rotation entirely (preserves any prior install that relied
// on unbounded logs + external rotation).
func slotLogRotation() (maxBytes int64, backups int) {
	return envInt64("QUENCHFORGE_LOG_MAX_BYTES", defaultSlotLogMaxBytes),
		envInt("QUENCHFORGE_LOG_BACKUPS", defaultSlotLogBackups)
}

func envInt64(key string, def int64) int64 {
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil || n < 0 {
		return def
	}
	return n
}

func envInt(key string, def int) int {
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return def
	}
	return n
}

// startSlot launches one llama-server with the configured model + Metal
// defaults. Wraps buildSlotArgs with the actual process supervision.
// maxLogBytes / logBackups come from the single slotLogRotation() call
// at the top of cmdServe — startSlot does NOT re-read them.
func startSlot(ctx context.Context, cfg config.Config, hwInfo hardware.Info, spec slotSpec, maxLogBytes int64, logBackups int, stderr io.Writer) (*supervisor.Slot, error) {
	bin, err := resolveLlamaBin(cfg.LlamaBin)
	if err != nil {
		return nil, err
	}
	modelPath, err := resolveModelPath(cfg.ModelsDir, spec.Model)
	if err != nil {
		return nil, err
	}

	args := buildSlotArgs(cfg, hwInfo, spec, modelPath)

	// Resolve tuning once (device-aware): drives the env and the restart
	// policy below so a dual-launch GPU instance gets its GPU env/respawn and
	// the CPU instance gets the minimal CPU env, even for an "auto" kind whose
	// policy device differs from the instance's actual device.
	tn := specTuning(cfg, hwInfo, spec)

	slot := supervisor.NewSlot(spec.Name)
	slot.BinPath = bin
	slot.LogDir = cfg.LogDir
	slot.PIDDir = cfg.PIDDir
	slot.Args = args
	slot.Env = envFromTuning(cfg, tn)
	slot.MaxLogBytes = maxLogBytes
	slot.LogBackups = logBackups

	// AMD-discrete embed/rerank slots need auto-respawn — the family-B
	// graph-compute buffer-corruption crash is non-deterministic and the
	// slot stays dead after SIGABRT until manual restart. Tuning module
	// owns the decision; we just translate AutoRespawn → RestartPolicy.
	if tn.AutoRespawn {
		slot.RestartPolicy = supervisor.PolicyExpBackoff
	}

	if err := slot.Start(ctx); err != nil {
		return nil, err
	}
	fmt.Fprintf(stderr, "quenchforge: spawned %s [%s] pid=%d log=%s maxBytes=%d backups=%d\n",
		bin, spec.Kind, slot.PID(),
		filepath.Join(cfg.LogDir, spec.Name+".log"), maxLogBytes, logBackups)
	return slot, nil
}

// ---------------------------------------------------------------------------
// migrate-from-ollama — symlink ~/.ollama/models GGUFs into models dir.
// ---------------------------------------------------------------------------

func cmdMigrate(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("migrate-from-ollama", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "list candidates without creating symlinks")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := config.Load()
	if err != nil {
		return err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	ollamaDir := filepath.Join(home, ".ollama", "models", "blobs")
	manifests := filepath.Join(home, ".ollama", "models", "manifests")

	if _, err := os.Stat(manifests); err != nil {
		return fmt.Errorf("no Ollama manifests at %s — is Ollama installed?", manifests)
	}

	if !*dryRun {
		if err := os.MkdirAll(cfg.ModelsDir, 0o755); err != nil {
			return err
		}
	}

	var linked, skipped int
	err = filepath.WalkDir(ollamaDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		// Ollama blob filenames look like sha256-<hex>. We don't know the
		// GGUF model name without parsing the manifests; for v0.1 we name
		// each symlink after the blob hash + .gguf, then let the user
		// rename. v0.2 walks the manifests to recover real model names.
		base := d.Name()
		if !strings.HasPrefix(base, "sha256-") {
			return nil
		}
		target := filepath.Join(cfg.ModelsDir, base+".gguf")
		if _, statErr := os.Lstat(target); statErr == nil {
			skipped++
			return nil
		}
		if *dryRun {
			fmt.Fprintf(stdout, "would link: %s -> %s\n", target, path)
			linked++
			return nil
		}
		if err := os.Symlink(path, target); err != nil {
			fmt.Fprintf(stderr, "symlink %s: %v\n", target, err)
			skipped++
			return nil
		}
		linked++
		return nil
	})
	if err != nil {
		return err
	}
	verb := "linked"
	if *dryRun {
		verb = "would link"
	}
	fmt.Fprintf(stdout, "quenchforge migrate: %d %s, %d skipped (already present)\n",
		linked, verb, skipped)
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "Note: blob hashes are the v0.1 filename. Rename to a model-friendly")
	fmt.Fprintln(stdout, "name (e.g. qwen2.5-7b-q4_k_m.gguf) by hand or wait for v0.2 manifest-aware")
	fmt.Fprintln(stdout, "migration which recovers Ollama-tagged names.")
	return nil
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// resolveSDBin finds sd-server (stable-diffusion.cpp HTTP example).
func resolveSDBin() (string, error) {
	for _, p := range []string{
		"./sd.cpp/build-arm64/bin/sd-server",
		"./sd.cpp/build-x86_64/bin/sd-server",
		"./sd.cpp/build-universal/bin/sd-server",
		"./sd.cpp/build/bin/sd-server",
		"/usr/local/bin/sd-server",
		"/opt/homebrew/bin/sd-server",
	} {
		if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Mode()&0o111 != 0 {
			return p, nil
		}
	}
	if p, err := lookPath("sd-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("sd-server not found — run scripts/build-sd.sh")
}

// resolveBarkBin finds the bark.cpp server example. Unlike sd-server, the
// upstream binary doesn't carry a distinctive name — it lives at
// examples/server/server. We accept both names for flexibility.
func resolveBarkBin() (string, error) {
	for _, p := range []string{
		"./bark.cpp/build-arm64/examples/server/server",
		"./bark.cpp/build-x86_64/examples/server/server",
		"./bark.cpp/build-universal/examples/server/server",
		"./bark.cpp/build/examples/server/server",
		"/usr/local/bin/bark-server",
		"/opt/homebrew/bin/bark-server",
	} {
		if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Mode()&0o111 != 0 {
			return p, nil
		}
	}
	if p, err := lookPath("bark-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("bark server not found — run scripts/build-bark.sh")
}

// resolveWhisperBin finds whisper-server, the patched companion binary
// from the whisper.cpp submodule. Search order mirrors resolveLlamaBin.
func resolveWhisperBin() (string, error) {
	tried := []string{}
	check := func(p string) (string, bool) {
		tried = append(tried, p)
		if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Mode()&0o111 != 0 {
			return p, true
		}
		return "", false
	}
	for _, p := range []string{
		"./whisper.cpp/build-arm64/bin/whisper-server",
		"./whisper.cpp/build-x86_64/bin/whisper-server",
		"./whisper.cpp/build-universal/bin/whisper-server",
		"./whisper.cpp/build/bin/whisper-server",
		"/usr/local/bin/whisper-server",
		"/opt/homebrew/bin/whisper-server",
	} {
		if r, ok := check(p); ok {
			return r, nil
		}
	}
	if p, err := lookPath("whisper-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("whisper-server not found (tried: %s) — run scripts/build-whisper.sh",
		strings.Join(tried, ", "))
}

// resolveLlamaBin finds the llama-server executable. Search order:
//
//  1. explicit path passed in (cfg.LlamaBin)
//  2. ./llama.cpp/build-*/bin/llama-server (post-build artifact)
//  3. /usr/local/bin/llama-server (Homebrew prefix on Intel Mac)
//  4. /opt/homebrew/bin/llama-server (Homebrew prefix on Apple Silicon)
//  5. PATH lookup
//
// Returns an error message that lists what was tried.
func resolveLlamaBin(hint string) (string, error) {
	tried := []string{}
	check := func(p string) (string, bool) {
		tried = append(tried, p)
		if st, err := os.Stat(p); err == nil && !st.IsDir() && st.Mode()&0o111 != 0 {
			return p, true
		}
		return "", false
	}
	if hint != "" {
		if p, ok := check(hint); ok {
			return p, nil
		}
	}
	for _, p := range []string{
		"./llama.cpp/build-arm64/bin/llama-server",
		"./llama.cpp/build-x86_64/bin/llama-server",
		"./llama.cpp/build-universal/bin/llama-server",
		"./llama.cpp/build/bin/llama-server",
		"/usr/local/bin/llama-server",
		"/opt/homebrew/bin/llama-server",
	} {
		if r, ok := check(p); ok {
			return r, nil
		}
	}
	// PATH lookup
	if p, err := lookPath("llama-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("llama-server not found (tried: %s)", strings.Join(tried, ", "))
}

// resolveModelPath turns a model name (with or without .gguf extension)
// into an absolute path under modelsDir. Falls through to the basename match
// if the exact path doesn't exist.
func resolveModelPath(modelsDir, name string) (string, error) {
	candidates := []string{
		filepath.Join(modelsDir, name),
		filepath.Join(modelsDir, name+".gguf"),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	// Last-ditch: walk and match basename
	var found string
	_ = filepath.WalkDir(modelsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		base := strings.TrimSuffix(d.Name(), ".gguf")
		if base == name {
			found = path
			return filepath.SkipAll
		}
		return nil
	})
	if found != "" {
		return found, nil
	}
	return "", fmt.Errorf("model %q not found under %s", name, modelsDir)
}

// lookPath is a thin wrapper so tests can monkey-patch the PATH search.
var lookPath = func(name string) (string, error) {
	return exec.LookPath(name)
}

// slotLine renders one slot's doctor row. When the slot's model is unset
// the line says "(opt-in: set $QUENCHFORGE_*_MODEL to enable)" so an
// operator can copy a known port and know exactly which env var to flip.
func slotLine(model string, port int) string {
	if model == "" {
		return fmt.Sprintf("(opt-in; port=%d)", port)
	}
	return fmt.Sprintf("model=%s port=%d", model, port)
}

// redactPath replaces the user's home dir with "~" when --redacted is set.
// Cheap PII reduction for paste-safe doctor output.
func redactPath(p string, redacted bool) string {
	if !redacted || p == "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return p
	}
	if strings.HasPrefix(p, home) {
		return "~" + strings.TrimPrefix(p, home)
	}
	return p
}
