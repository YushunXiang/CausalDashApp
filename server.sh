#!/bin/bash

echo "🚀 Starting Debug Visualization Server"
echo "======================================"

cd "$(dirname "$0")"

python server/debug_server.py \
    --host localhost \
    --port 9001 \
    --images resources/balloon.png resources/weight.png \
    --interval 1.0 \
    "$@"