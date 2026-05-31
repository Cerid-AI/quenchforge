// Command quenchforge — `install` subcommand.
//
// Drops the LaunchAgent plist into ~/Library/LaunchAgents/ with
// the operator's $USER substituted into the REPLACE_ME placeholders,
// then prints next-step instructions for `launchctl bootstrap`.
//
// macOS-only. On Linux/BSD it returns a clear error rather than
// silently no-oping.
package main

import (
	"bytes"
	_ "embed"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

//go:embed plist_template.plist
var plistTemplate []byte

//go:embed prestart-guard.sh
var prestartGuard []byte

const plistFilename = "com.cerid.quenchforge.plist"

// prestartGuardRelPath is where the guard is written under the operator's
// HOME. The generated plist's ProgramArguments[0] points here (via the
// REPLACE_ME → $USER substitution), so the two must stay in sync.
var prestartGuardRelPath = filepath.Join(".config", "quenchforge", "prestart-guard.sh")

func cmdInstall(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	force := fs.Bool("force", false, "overwrite an existing plist")
	skipUserSub := fs.Bool("skip-user-substitution", false,
		"leave REPLACE_ME placeholders unchanged (for operators who want to edit by hand)")
	printPath := fs.Bool("print-path", false,
		"print the resolved target path and exit without writing")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "Usage: quenchforge install [--force] [--skip-user-substitution] [--print-path]")
		fmt.Fprintln(stderr)
		fmt.Fprintln(stderr, "Install the LaunchAgent plist into ~/Library/LaunchAgents/.")
		fmt.Fprintln(stderr, "Substitutes the operator's $USER into the REPLACE_ME placeholders by default.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}

	if runtime.GOOS != "darwin" {
		return fmt.Errorf("install: macOS only (current platform: %s/%s)",
			runtime.GOOS, runtime.GOARCH)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("install: cannot determine HOME: %w", err)
	}
	targetDir := filepath.Join(home, "Library", "LaunchAgents")
	targetPath := filepath.Join(targetDir, plistFilename)

	if *printPath {
		fmt.Fprintln(stdout, targetPath)
		return nil
	}

	if _, err := os.Stat(targetPath); err == nil {
		if !*force {
			return fmt.Errorf(
				"install: %s already exists. Pass --force to overwrite, "+
					"or remove it manually with `launchctl bootout gui/$(id -u)/com.cerid.quenchforge && rm %s`",
				targetPath, targetPath)
		}
		fmt.Fprintf(stdout, "Overwriting existing plist at %s\n", targetPath)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("install: stat %s: %w", targetPath, err)
	}

	data := plistTemplate
	if !*skipUserSub {
		user := os.Getenv("USER")
		if user == "" {
			return fmt.Errorf("install: USER environment variable is empty; " +
				"set USER or pass --skip-user-substitution to leave REPLACE_ME placeholders unchanged")
		}
		data = bytes.ReplaceAll(plistTemplate, []byte("REPLACE_ME"), []byte(user))
	}

	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return fmt.Errorf("install: mkdir %s: %w", targetDir, err)
	}

	if err := os.WriteFile(targetPath, data, 0o644); err != nil {
		return fmt.Errorf("install: write %s: %w", targetPath, err)
	}

	// Write the prestart guard the plist's ProgramArguments[0] points at.
	// It reclaims port 11434 from a squatter (e.g. Ollama) before exec'ing
	// `quenchforge serve`. Executable; lives under the operator's HOME so
	// the REPLACE_ME → $USER substitution in the plist resolves to it.
	guardPath := filepath.Join(home, prestartGuardRelPath)
	if err := os.MkdirAll(filepath.Dir(guardPath), 0o755); err != nil {
		return fmt.Errorf("install: mkdir %s: %w", filepath.Dir(guardPath), err)
	}
	if err := os.WriteFile(guardPath, prestartGuard, 0o755); err != nil {
		return fmt.Errorf("install: write prestart guard %s: %w", guardPath, err)
	}

	fmt.Fprintf(stdout, "Installed LaunchAgent at %s (%d bytes)\n", targetPath, len(data))
	fmt.Fprintf(stdout, "Installed prestart port guard at %s\n", guardPath)
	if !*skipUserSub {
		fmt.Fprintf(stdout, "  Substituted REPLACE_ME → %s\n", os.Getenv("USER"))
	}
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "Next steps:")
	fmt.Fprintln(stdout, "  1. Inspect model env vars against your installed GGUFs:")
	fmt.Fprintf(stdout, "       less %s\n", targetPath)
	fmt.Fprintln(stdout, "       quenchforge list                 # show locally cached models")
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "  2. Bootstrap the service (atomic plist load):")
	fmt.Fprintf(stdout, "       launchctl bootstrap gui/$(id -u) %s\n", targetPath)
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "  3. Verify it's serving:")
	fmt.Fprintln(stdout, "       curl http://127.0.0.1:11434/")
	fmt.Fprintln(stdout)
	fmt.Fprintln(stdout, "To uninstall later:")
	fmt.Fprintf(stdout, "  launchctl bootout gui/$(id -u)/com.cerid.quenchforge && rm %s\n", targetPath)

	return nil
}
