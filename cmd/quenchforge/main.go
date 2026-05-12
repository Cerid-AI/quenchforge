// Command quenchforge is the user-facing CLI and daemon entrypoint.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
	"github.com/cerid-ai/quenchforge/internal/hardware"
	"github.com/cerid-ai/quenchforge/internal/supervisor"
)

// Version is injected at build time via:
//
//	go build -ldflags "-X main.Version=$(git describe --tags --always)"
//
// goreleaser handles this in CI. Local dev builds carry the zero value.
var (
	Version   = "0.0.0-dev"
	Commit    = "unknown"
	BuildDate = "unknown"
)

const rootUsage = `quenchforge — local inference for Mac + AMD discrete GPU

Usage:
    quenchforge <command> [arguments]

Commands:
    serve              Start the HTTP gateway (Ollama + OpenAI compatible).
    doctor             Print a hardware-and-environment report for bug triage.
    migrate-from-ollama  Symlink ~/.ollama/models/ GGUFs into the quenchforge cache.
    version            Print version, commit, and build date.
    help               Show this message.

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
	case "migrate-from-ollama":
		return cmdMigrate(args[1:], stdout, stderr)
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

	fmt.Fprintln(stdout, "Paste this output verbatim into bug reports.")
	return nil
}

// ---------------------------------------------------------------------------
// serve — gateway + (optional) supervised llama-server slot.
// ---------------------------------------------------------------------------

func cmdServe(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	listen := fs.String("listen", "", "override listen address (env: QUENCHFORGE_LISTEN_ADDR)")
	noSlot := fs.Bool("no-slot", false, "start the gateway only; don't supervise a llama-server slot")
	model := fs.String("model", "", "GGUF model name (under models dir) to load in the chat slot; "+
		"empty = read from QUENCHFORGE_DEFAULT_MODEL")
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
	if err := cfg.EnsureDirs(); err != nil {
		return err
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

	// Optional supervised slot
	var slot *supervisor.Slot
	if !*noSlot {
		modelName := *model
		if modelName == "" {
			modelName = cfg.DefaultModel
		}
		var err error
		slot, err = startChatSlot(ctx, cfg, modelName, stderr)
		if err != nil {
			fmt.Fprintf(stderr,
				"quenchforge: warning: chat slot not started: %v\n"+
					"  Gateway is up but /api/chat will return 503.\n"+
					"  Run `quenchforge doctor` for the binary path and model registry.\n", err)
		} else {
			fmt.Fprintf(stdout, "quenchforge: chat slot pid=%d model=%s\n",
				slot.PID(), modelName)
			// Point the gateway at the slot's local URL. v0.1 hard-codes the
			// slot port; v0.2 reads the actual bind from llama-server's stderr.
			_ = g.SetUpstream("http://127.0.0.1:11500")
		}
	}

	// Wait for shutdown signal
	<-ctx.Done()
	fmt.Fprintln(stdout, "quenchforge: shutdown signal received")

	// Tear down: slot first, then gateway, with grace timeouts so a hung
	// llama-server can't block our exit indefinitely.
	if slot != nil {
		if err := slot.Stop(5 * time.Second); err != nil {
			fmt.Fprintf(stderr, "quenchforge: slot stop: %v\n", err)
		}
	}
	if err := g.Stop(2 * time.Second); err != nil {
		fmt.Fprintf(stderr, "quenchforge: gateway stop: %v\n", err)
	}
	return nil
}

// startChatSlot launches llama-server with the configured model and Metal
// defaults. The slot binds to 127.0.0.1:11500 (one above 11434 so the
// gateway port is never in conflict with its child).
func startChatSlot(ctx context.Context, cfg config.Config, modelName string, stderr io.Writer) (*supervisor.Slot, error) {
	bin, err := resolveLlamaBin(cfg.LlamaBin)
	if err != nil {
		return nil, err
	}
	modelPath, err := resolveModelPath(cfg.ModelsDir, modelName)
	if err != nil {
		return nil, err
	}

	slot := supervisor.NewSlot("chat")
	slot.BinPath = bin
	slot.LogDir = cfg.LogDir
	slot.PIDDir = cfg.PIDDir
	slot.Args = []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", "11500",
		"--ctx-size", fmt.Sprintf("%d", cfg.MaxContext),
	}
	slot.Env = []string{
		fmt.Sprintf("GGML_METAL_N_CB=%d", cfg.MetalNCB),
		"GGML_METAL_FORCE_PRIVATE=1", // honored by patch 0001 once applied
	}
	if err := slot.Start(ctx); err != nil {
		return nil, err
	}
	fmt.Fprintf(stderr, "quenchforge: spawned %s pid=%d log=%s\n",
		bin, slot.PID(), filepath.Join(cfg.LogDir, "chat.log"))
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
