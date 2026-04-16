#!/bin/bash
# Build and run the _ANEChainingRequest probe
set -e
cd "$(dirname "$0")"

echo "==> Compiling probe_chaining.m ..."
xcrun --sdk macosx clang \
    -fmodules \
    -framework Foundation \
    -framework IOSurface \
    -o probe_chaining \
    probe_chaining.m

echo "==> Running ..."
echo ""
./probe_chaining 2>&1
