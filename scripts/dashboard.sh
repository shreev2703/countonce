#!/usr/bin/env bash
watch -n 2 '
echo "=== COUNTONCE ORCHESTRATOR DASHBOARD ==="
echo "Status: $(tail -n 1 ~/countonce/logs/overnight_*/orchestrator.log 2>/dev/null | grep -v "^==>" | tail -1)"
echo ""
echo "=== TASK QUEUE ==="
cat ~/countonce/TASKS_OVERNIGHT.md
echo ""
echo "=== LATEST AGENT ACTIVITY & USAGE ==="
LATEST=$(ls -t ~/countonce/logs/overnight_*/*.log 2>/dev/null | grep -v orchestrator | head -1)
if [ -n "$LATEST" ]; then
  grep -oE "\"(name|model|input_tokens|output_tokens|total_cost_usd)\":[^,}]+" "$LATEST" | tail -n 15
fi
'
