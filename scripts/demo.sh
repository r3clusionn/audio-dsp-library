#!/usr/bin/env bash
# A short session with the adsp command, for the README screenshot.
# Usage: scripts/demo.sh   (after cargo build --release)
set -euo pipefail
cd "$(dirname "$0")/.."
B=target/release/adsp
D=target/demo
mkdir -p "$D"
run() { echo "\$ adsp $*"; "$B" "$@"; echo; }
"$B" gen "$D/tone.wav" sine --freq 997 --seconds 2 --rate 44100 --level -6 > /dev/null
run info "$D/tone.wav"
run resample "$D/tone.wav" "$D/tone48.wav" 48000
run info "$D/tone48.wav"
"$B" gen "$D/sweep.wav" sweep --freq 20 --to 20000 --seconds 4 > /dev/null
run filter "$D/sweep.wav" "$D/hp.wav" highpass 2000 --order 8
echo "\$ adsp spectrum $D/hp.wav --per-octave 1"
"$B" spectrum "$D/hp.wav" --per-octave 1
