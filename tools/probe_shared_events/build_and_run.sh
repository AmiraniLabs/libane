#!/bin/bash
set -e
cd "$(dirname "$0")"
xcrun --sdk macosx clang \
    -fmodules \
    -framework Foundation \
    -framework IOSurface \
    -framework Metal \
    -o probe_shared_events \
    probe_shared_events.m
./probe_shared_events
