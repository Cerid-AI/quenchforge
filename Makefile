# Quenchforge — canonical build entry point
#
# Local builds get version + commit + build-date stamped via ldflags so
# `quenchforge --version` and `quenchforge doctor` report something
# useful instead of the package-default `0.3.x-dev (unknown)`.
#
# CI / release builds use the same ldflag pattern via .goreleaser.yaml
# — see the `ldflags:` section there. The values are slightly different
# (goreleaser uses {{.Version}} = the tag with no v-prefix; this Makefile
# uses git-describe so dev builds get a useful suffix).

PREFIX ?= /usr/local

# Build-time stamps. Falls back gracefully outside a git checkout (e.g.,
# a Homebrew bottle that's vendoring a source tarball).
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "0.3.4-dev")
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

LDFLAGS := -ldflags "\
	-X main.Version=$(VERSION) \
	-X main.Commit=$(COMMIT) \
	-X main.BuildDate=$(BUILD_DATE)"

.PHONY: help
help:
	@echo "Quenchforge build targets:"
	@echo "  make build               — patch submodules, build llama-server + quenchforge"
	@echo "  make build-go            — build only the Go binaries (assumes patched llama-server present)"
	@echo "  make install             — copy binaries to $(PREFIX)/bin (sudo may be needed)"
	@echo "  make test                — go test ./..."
	@echo "  make lint                — go vet + gofmt -l (CI parity)"
	@echo "  make patches             — apply patch series to submodules"
	@echo "  make clean               — remove build outputs"
	@echo
	@echo "Version stamp at next build:"
	@echo "  Version    $(VERSION)"
	@echo "  Commit     $(COMMIT)"
	@echo "  BuildDate  $(BUILD_DATE)"

.PHONY: patches
patches:
	bash scripts/apply-patches.sh

.PHONY: build
build: patches
	bash scripts/build-llama.sh
	$(MAKE) build-go

.PHONY: build-go
build-go:
	@mkdir -p bin
	go build $(LDFLAGS) -o bin/quenchforge ./cmd/quenchforge
	go build -o bin/quenchforge-preflight ./cmd/quenchforge-preflight
	@echo
	@echo "Built: bin/quenchforge $(VERSION) ($(COMMIT))"

.PHONY: install
install: build-go
	install -m 0755 bin/quenchforge $(PREFIX)/bin/quenchforge
	install -m 0755 bin/quenchforge-preflight $(PREFIX)/bin/quenchforge-preflight
	@echo
	@echo "Installed to $(PREFIX)/bin/"
	@$(PREFIX)/bin/quenchforge version 2>/dev/null | head -3

.PHONY: test
test:
	go test ./...

.PHONY: lint
lint:
	@unformatted=$$(gofmt -l cmd internal); \
	if [ -n "$$unformatted" ]; then \
		echo "::error::Unformatted Go files found:"; \
		echo "$$unformatted"; \
		exit 1; \
	fi
	go vet ./...

.PHONY: clean
clean:
	rm -rf bin/
	rm -rf llama.cpp/build-*
