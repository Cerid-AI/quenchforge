// Command quenchforge is the user-facing CLI and daemon entrypoint.
//
// This file is the scaffold form of the entrypoint: only `version`, `doctor`,
// and `help` are wired. The supervisor, gateway, and slot lifecycle land in
// MVP week 5 per the plan.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"time"
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
    doctor     Print a hardware-and-environment report suitable for bug triage.
    version    Print version, commit, and build date.
    help       Show this message.

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

func cmdDoctor(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	redacted := fs.Bool("redacted", false, "produce a paste-safe report (no usernames, no paths beyond ~)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	// MVP-stage scaffold output. The real implementation lives in
	// internal/hardware/detect_darwin.go and is wired in MVP week 3.
	fmt.Fprintln(stdout, "quenchforge doctor")
	fmt.Fprintln(stdout, "==================")
	fmt.Fprintf(stdout, "  version:        %s (%s)\n", Version, Commit)
	fmt.Fprintf(stdout, "  build date:     %s\n", BuildDate)
	fmt.Fprintf(stdout, "  runtime:        go %s on %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(stdout, "  generated:      %s\n", time.Now().UTC().Format(time.RFC3339))
	fmt.Fprintln(stdout)

	if runtime.GOOS != "darwin" {
		fmt.Fprintln(stdout, "  status:         UNSUPPORTED (quenchforge is macOS-only)")
		return nil
	}

	fmt.Fprintln(stdout, "  status:         scaffold — hardware probe not wired yet")
	fmt.Fprintln(stdout, "  next milestone: internal/hardware/detect_darwin.go (MVP week 3)")
	if *redacted {
		fmt.Fprintln(stdout, "  --redacted:     acknowledged (no-op until real probe lands)")
	}
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "Paste this output verbatim into bug reports.")

	return nil
}
