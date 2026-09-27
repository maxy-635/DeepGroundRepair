#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: bash baselines/run_main.sh <B1|B2|B3|B4> <api|hf>"
    exit 1
fi

BASELINE="$1"
MODE="$2"

case "$BASELINE" in
    B1|B2|B3|B4)
        ;;
    *)
        echo "Invalid baseline: $BASELINE"
        echo "Expected one of: B1, B2, B3, B4"
        exit 1
        ;;
esac

case "$MODE" in
    api)
        SCRIPT_SUFFIX="api"
        ;;
    hf)
        SCRIPT_SUFFIX="hf"
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Expected one of: api, hf"
        exit 1
        ;;
esac


SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TARGET_SCRIPT="$SCRIPT_DIR/$BASELINE/run_$SCRIPT_SUFFIX.sh"

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Missing target script: $TARGET_SCRIPT"
    exit 1
fi

echo "Running [$BASELINE] with [$MODE] mode......"
bash "$TARGET_SCRIPT"