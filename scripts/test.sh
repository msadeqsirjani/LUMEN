#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 basicsr/test.py -opt "$@"
