#!/bin/bash
set -e
cd "$(dirname "$0")"
xcrun --sdk macosx clang \
    -fmodules \
    -framework Foundation \
    -framework IOSurface \
    -o probe_offset \
    probe_offset.m
./probe_offset
