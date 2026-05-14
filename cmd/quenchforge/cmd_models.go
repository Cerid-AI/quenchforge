// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Model-management subcommands: `pull`, `list`, `rm`.
//
// These close the biggest UX gap vs Ollama for new operators. Pre-v0.4,
// the only path to a working model was manual GGUF placement under
// ~/.quenchforge/models/ or `quenchforge migrate-from-ollama` (which
// only worked for operators already running Ollama). Now:
//
//	quenchforge pull llama3.2:3b                    # via catalog alias
//	quenchforge pull bartowski/Repo-GGUF:Q4_K_M     # explicit
//	quenchforge list                                 # what's installed
//	quenchforge rm llama3.2:3b                       # remove

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/registry"
)

const pullUsage = `quenchforge pull — download a GGUF from HuggingFace

Usage:
    quenchforge pull [flags] <spec>

Spec forms:
    <alias>                            # catalog: see ` + "`quenchforge pull --list`" + `
    <user>/<repo>:<quant>              # e.g. bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
    <user>/<repo>/<file>.gguf          # explicit filename

Flags:
    --list                             # print the catalog of well-tested aliases
    --no-progress                      # suppress the progress bar
    --models-dir <path>                # override the install dest (default: $QUENCHFORGE_MODELS_DIR or ~/.quenchforge/models)

Authentication for private HF repos: set HF_TOKEN before invocation.
`

func cmdPull(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	fs.SetOutput(stderr)
	listCatalog := fs.Bool("list", false, "print the catalog of well-tested aliases and exit")
	noProgress := fs.Bool("no-progress", false, "suppress the progress bar")
	modelsDirFlag := fs.String("models-dir", "", "override the install destination")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *listCatalog {
		return printCatalog(stdout)
	}

	if fs.NArg() != 1 {
		fmt.Fprint(stderr, pullUsage)
		return fmt.Errorf("pull: exactly one spec argument required")
	}
	specArg := fs.Arg(0)

	spec, err := registry.ParseSpec(specArg)
	if err != nil {
		return err
	}

	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	modelsDir := cfg.ModelsDir
	if *modelsDirFlag != "" {
		modelsDir = *modelsDirFlag
	}

	client := registry.New(modelsDir)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	fmt.Fprintf(stdout, "quenchforge pull: %s\n", specArg)
	fmt.Fprintf(stdout, "  repo:       %s\n", spec.Repo)
	fmt.Fprintf(stdout, "  file match: %s\n", spec.FileMatch)
	fmt.Fprintf(stdout, "  dest:       %s/%s.gguf\n", modelsDir, spec.LocalName)
	fmt.Fprintln(stdout)

	var progressFn registry.ProgressFn
	if !*noProgress {
		progressFn = newProgressPrinter(stdout)
	}

	start := time.Now()
	finalPath, err := client.Pull(ctx, spec, progressFn)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	info, _ := os.Stat(finalPath)
	if info != nil {
		mbps := float64(info.Size()) / (1 << 20) / elapsed.Seconds()
		fmt.Fprintf(stdout, "\ndone: %s (%.2f GB in %s, %.1f MB/s)\n",
			finalPath, float64(info.Size())/(1<<30), elapsed.Truncate(time.Second), mbps)
	} else {
		fmt.Fprintf(stdout, "\ndone: %s\n", finalPath)
	}
	return nil
}

func cmdList(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelsDirFlag := fs.String("models-dir", "", "override the models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	modelsDir := cfg.ModelsDir
	if *modelsDirFlag != "" {
		modelsDir = *modelsDirFlag
	}

	models, err := registry.List(modelsDir)
	if err != nil {
		return err
	}
	if len(models) == 0 {
		fmt.Fprintf(stdout, "no GGUFs cached under %s\n", modelsDir)
		fmt.Fprintln(stdout, "hint: try `quenchforge pull llama3.2:3b` or `quenchforge pull --list` for the catalog")
		return nil
	}

	// Sort by name for stable output.
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })

	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tSIZE\tMODIFIED\tTYPE")
	for _, m := range models {
		kind := "file"
		if m.Symlink {
			kind = "symlink"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n",
			m.Name,
			humanSize(m.SizeBytes),
			m.ModifiedAt.Format("2006-01-02 15:04"),
			kind,
		)
	}
	return tw.Flush()
}

func cmdRm(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("rm", flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelsDirFlag := fs.String("models-dir", "", "override the models directory")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if fs.NArg() == 0 {
		return fmt.Errorf("rm: at least one model name required (see `quenchforge list`)")
	}

	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	modelsDir := cfg.ModelsDir
	if *modelsDirFlag != "" {
		modelsDir = *modelsDirFlag
	}

	var errors []string
	for _, name := range fs.Args() {
		// Tolerate `.gguf` suffix in case the user copy-pasted from `list`.
		name = strings.TrimSuffix(name, ".gguf")
		if err := registry.Remove(modelsDir, name); err != nil {
			errors = append(errors, err.Error())
			continue
		}
		fmt.Fprintf(stdout, "removed: %s\n", name)
	}
	if len(errors) > 0 {
		return fmt.Errorf("%s", strings.Join(errors, "; "))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// newProgressPrinter returns a ProgressFn that overwrites the same line
// on stdout with a textual progress bar. Calls are rate-limited inside
// the registry package to ~1/MB, so we don't need additional throttling.
func newProgressPrinter(out io.Writer) registry.ProgressFn {
	return func(done, total int64) {
		if total <= 0 {
			fmt.Fprintf(out, "\r  downloading: %s", humanSize(done))
			return
		}
		pct := float64(done) / float64(total) * 100
		// 30-character bar
		filled := int(pct / 100 * 30)
		bar := strings.Repeat("█", filled) + strings.Repeat("░", 30-filled)
		fmt.Fprintf(out, "\r  [%s] %5.1f%%  %s / %s",
			bar, pct, humanSize(done), humanSize(total))
	}
}

func humanSize(n int64) string {
	const (
		_  = iota
		KB = 1 << (10 * iota)
		MB
		GB
		TB
	)
	f := float64(n)
	switch {
	case n < KB:
		return fmt.Sprintf("%d B", n)
	case n < MB:
		return fmt.Sprintf("%.1f KB", f/KB)
	case n < GB:
		return fmt.Sprintf("%.1f MB", f/MB)
	case n < TB:
		return fmt.Sprintf("%.2f GB", f/GB)
	default:
		return fmt.Sprintf("%.2f TB", f/TB)
	}
}

func printCatalog(out io.Writer) error {
	tw := tabwriter.NewWriter(out, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "ALIAS\tREPO\tQUANT\tVRAM\tNOTES")
	for _, e := range registry.Catalog() {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d GB\t%s\n",
			e.Alias, e.Repo, e.FileMatch, e.VRAMGB, e.Notes)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	fmt.Fprintln(out)
	fmt.Fprintln(out, "These aliases are the curated landing pad. Operators who need other quants")
	fmt.Fprintln(out, "or models can pass the full `<user>/<repo>:<quant>` spec directly to `pull`.")
	return nil
}
