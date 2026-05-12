# typed: false
# frozen_string_literal: true

# This Formula file is the SCAFFOLD that goreleaser regenerates on every
# release. It lives in this repository so the tap structure is documented;
# the LIVE formula consumed by `brew install` lives at
# https://github.com/Cerid-AI/homebrew-tap/blob/main/Formula/quenchforge.rb
# and is overwritten by the brews: section of .goreleaser.yaml on release.
#
# The version/sha256 fields below are placeholders. goreleaser substitutes
# real values via its template engine — see `goreleaser release --snapshot
# --skip=publish` for what gets rendered.

class Quenchforge < Formula
  desc "ggml-on-AMD-Mac correctness — chat, embedding, reranker, Whisper transcription"
  homepage "https://github.com/Cerid-AI/quenchforge"
  license "Apache-2.0"
  version "0.3.0-dev"

  # Hardware constraint — Quenchforge is macOS-only by design, and the
  # patches only matter on Intel Mac + AMD discrete or Apple Silicon. We
  # don't refuse install on supported macOS configs that lack a discrete
  # GPU (e.g. MacBook Air); they get an unaccelerated fallback and a
  # `quenchforge doctor` warning.
  depends_on :macos => :sonoma

  on_macos do
    on_arm do
      url "https://github.com/Cerid-AI/quenchforge/releases/download/v#{version}/quenchforge_#{version}_darwin_arm64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
    on_intel do
      url "https://github.com/Cerid-AI/quenchforge/releases/download/v#{version}/quenchforge_#{version}_darwin_amd64.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
  end

  def install
    bin.install "quenchforge"
    bin.install "quenchforge-preflight"
    # The patches/ directory ships in the tarball so operators with the
    # binary install can still run `quenchforge-preflight` and read the
    # provenance — patches are applied to llama.cpp out-of-band on build.
    pkgshare.install Dir["patches/*"]
    pkgshare.install "LICENSE", "NOTICE", "README.md", "SECURITY.md"
  end

  # `brew services` integration — generates the launchd plist automatically
  # via the Homebrew service block. Modern best-practice; do not ship a
  # separate .plist file alongside.
  service do
    run [opt_bin/"quenchforge", "serve"]
    keep_alive true
    log_path "#{var}/log/quenchforge/serve.log"
    error_log_path "#{var}/log/quenchforge/serve.err.log"
    working_dir var
    environment_variables QUENCHFORGE_LOG_DIR: "#{var}/log/quenchforge"
  end

  def post_install
    (var/"log/quenchforge").mkpath
  end

  def caveats
    <<~EOS
      Quenchforge is installed but the patched llama.cpp binary is NOT bundled —
      operators on the AMD Mac path must build llama-server from source:

        git clone --recursive https://github.com/Cerid-AI/quenchforge /tmp/qf-src
        cd /tmp/qf-src
        bash scripts/apply-patches.sh
        bash scripts/build-llama.sh
        sudo install -m 0755 llama.cpp/build-*/bin/llama-server \\
          /usr/local/bin/llama-server

      Start the service:
        brew services start quenchforge
        quenchforge doctor

      First-launch notes:
        - On Sonoma+, if you set QUENCHFORGE_ADVERTISE_MDNS=true, macOS will
          show a "find devices on your local network" prompt for mDNSResponder.
          Approve it for cerid-ai LAN autodiscovery; decline if you only want
          loopback access.
    EOS
  end

  test do
    out = shell_output("#{bin}/quenchforge version")
    assert_match "quenchforge", out
    assert_match "darwin", out
    pf_out = shell_output("#{bin}/quenchforge-preflight 2>&1", 0)
    assert_match "status=", pf_out
  end
end
