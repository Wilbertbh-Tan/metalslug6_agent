#!/bin/bash
# Monitor training logs for issues like stalls, errors, and "n/a" values.
# Usage:
#   ./scripts/monitor_training.sh                     # tail latest run's log (local)
#   ./scripts/monitor_training.sh path/to/log         # tail specific log
#   ./scripts/monitor_training.sh --tensorboard       # start TensorBoard on outputs/runs
#   ./scripts/monitor_training.sh --docker [NAME]     # tail docker logs (default: mslug6)

if [ "$1" = "--docker" ]; then
    CONTAINER="${2:-mslug6}"
    echo "Tailing Docker container: $CONTAINER (Ctrl+C to stop)"
    exec docker logs -f "$CONTAINER"
fi

if [ "$1" = "--tensorboard" ]; then
    echo "Starting TensorBoard on outputs/runs (all runs)..."
    echo "Open http://localhost:6006 in your browser."
    exec tensorboard --logdir outputs/runs --port 6006
fi

# Default: latest run's training_stdout.log (outputs/runs/PPO_N/logs/training_stdout.log)
if [ -z "$1" ]; then
    latest=""
    for f in outputs/runs/*/logs/training_stdout.log; do
        [ -f "$f" ] || continue
        if [ -z "$latest" ] || [ "$f" -nt "$latest" ]; then
            latest="$f"
        fi
    done
    LOG_FILE="${latest:-outputs/logs/training_stdout.log}"
else
    LOG_FILE="$1"
fi
STALL_KEYWORD="[STALL]"
ERROR_KEYWORDS="error|Error|ERROR|exception|Exception|Traceback|failed|Failed|FAILED"
NA_KEYWORDS="n/a|N/A|NA|None|none"
WARNING_KEYWORDS="WARNING|Warning|warning"

echo "Monitoring training log: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo "=================================="
echo ""

# Function to check if file exists and is readable
if [ ! -f "$LOG_FILE" ]; then
    echo "Warning: Log file not found: $LOG_FILE"
    echo "Waiting for file to be created..."
    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done
    echo "Log file detected, starting monitoring..."
fi

# Monitor in real-time
tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # Check for stalls
    if echo "$line" | grep -qi "$STALL_KEYWORD"; then
        echo -e "\033[1;31m[STALL DETECTED]\033[0m $line"
    fi
    # Check for errors
    if echo "$line" | grep -qiE "$ERROR_KEYWORDS"; then
        echo -e "\033[1;31m[ERROR]\033[0m $line"
    fi
    # Check for warnings
    if echo "$line" | grep -qiE "$WARNING_KEYWORDS"; then
        echo -e "\033[1;33m[WARNING]\033[0m $line"
    fi
    # Check for n/a values (might indicate missing data)
    if echo "$line" | grep -qiE "$NA_KEYWORDS"; then
        # Only highlight if it's in a monitor/episode line (not just any line with "n/a")
        if echo "$line" | grep -qiE "\[monitor\]|\[episode\]"; then
            echo -e "\033[1;33m[MISSING DATA]\033[0m $line"
        fi
    fi
    # Show all monitor lines (training progress)
    if echo "$line" | grep -qiE "\[monitor\]|\[episode\]"; then
        echo "$line"
    fi
done
