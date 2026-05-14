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
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/discovery"
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
	Version   = "0.4.0-dev"
	Commit    = "unknown"
	BuildDate = "unknown"
)

const rootUsage = `quenchforge — local inference for Mac + AMD discrete GPU

Usage:
    quenchforge <command> [arguments]

Commands:
    serve              Start the HTTP gateway (Ollama + OpenAI compatible).
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
		}, stderr)
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
		if cfg.EmbedModel != "" {
			s, err := startSlot(ctx, cfg, hwInfo, slotSpec{
				Kind:  gateway.KindEmbed,
				Name:  "embed",
				Model: cfg.EmbedModel,
				Port:  cfg.EmbedPort,
				// --embedding flips llama-server into pooled-embedding mode.
				// --pooling cls is the standard for most BERT-style embedders;
				// callers using mean-pooling models can override via config.
				ExtraArgs: []string{"--embedding", "--pooling", "cls"},
			}, stderr)
			if err != nil {
				fmt.Fprintf(stderr,
					"quenchforge: warning: embed slot not started: %v\n"+
						"  /api/embeddings will return 503.\n", err)
			} else {
				slots[gateway.KindEmbed] = s
				fmt.Fprintf(stdout, "quenchforge: embed slot pid=%d model=%s port=%d\n",
					s.PID(), cfg.EmbedModel, cfg.EmbedPort)
				_ = g.SetUpstream(gateway.KindEmbed,
					fmt.Sprintf("http://127.0.0.1:%d", cfg.EmbedPort))
			}
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
			}, stderr)
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
}

// buildSlotArgs constructs the llama-server command-line arguments for
// one supervised slot. Pure function (no I/O, no globals) so it can be
// unit-tested without spawning llama-server.
//
// AMD-discrete chat-slot flags:
//
//	--flash-attn off     The default `--flash-attn auto` correctly
//	                     determines FA can't run on AMD MTL0 (the
//	                     simdgroup-reduction patch disables the ops
//	                     FA needs) but instead of disabling FA
//	                     outright it schedules the FA tensor on CPU
//	                     each decode step, ferrying tensors GPU↔CPU
//	                     per token. Forcing `off` uses standard
//	                     (slower per kernel, but GPU-resident)
//	                     attention.
//
//	--cache-ram 0        Disables the server-side LCP-similarity slot
//	                     cache. The crash signature is in
//	                     `prompt_save` → `state_seq_get_data` →
//	                     `ggml_metal_buffer_get_tensor(buf_dst=NULL)`
//	                     on the 2nd chat with LCP similarity > 10%.
//	                     --cache-ram 0 disables the path entirely.
//
//	--no-cache-prompt    Companion to --cache-ram 0 — disables the
//	                     per-slot prompt cache so the LCP-similarity
//	                     path can't fire from a per-slot trigger in
//	                     a future llama.cpp release. The two flags
//	                     together belt-and-suspenders the entire
//	                     prompt-cache surface.
//
// Embed / rerank slots don't decode autoregressively and don't touch
// the server-side cache, so they keep the upstream defaults regardless
// of profile. Apple Silicon and unknown profiles also keep the
// upstream defaults across all slot kinds.
func buildSlotArgs(cfg config.Config, hwInfo hardware.Info, spec slotSpec, modelPath string) []string {
	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", spec.Port),
		"--ctx-size", fmt.Sprintf("%d", cfg.MaxContext),
	}
	args = append(args, spec.ExtraArgs...)

	if spec.Kind == gateway.KindChat && hwInfo.IsAMDDiscrete() {
		args = append(args,
			"--flash-attn", "off",
			"--cache-ram", "0",
			"--no-cache-prompt",
		)
	}
	return args
}

// startSlot launches one llama-server with the configured model + Metal
// defaults. Wraps buildSlotArgs with the actual process supervision.
func startSlot(ctx context.Context, cfg config.Config, hwInfo hardware.Info, spec slotSpec, stderr io.Writer) (*supervisor.Slot, error) {
	bin, err := resolveLlamaBin(cfg.LlamaBin)
	if err != nil {
		return nil, err
	}
	modelPath, err := resolveModelPath(cfg.ModelsDir, spec.Model)
	if err != nil {
		return nil, err
	}

	args := buildSlotArgs(cfg, hwInfo, spec, modelPath)

	slot := supervisor.NewSlot(spec.Name)
	slot.BinPath = bin
	slot.LogDir = cfg.LogDir
	slot.PIDDir = cfg.PIDDir
	slot.Args = args
	slot.Env = []string{
		fmt.Sprintf("GGML_METAL_N_CB=%d", cfg.MetalNCB),
	}
	if err := slot.Start(ctx); err != nil {
		return nil, err
	}
	fmt.Fprintf(stderr, "quenchforge: spawned %s [%s] pid=%d log=%s\n",
		bin, spec.Kind, slot.PID(), filepath.Join(cfg.LogDir, spec.Name+".log"))
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
