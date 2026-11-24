#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="$HOME/countonce"
TASKS_FILE="$REPO_DIR/TASKS_OVERNIGHT.md"
HANDOFF_FILE="/mnt/c/Users/super_nga/second-brain/countonce-handoff.md"
LOG_DIR="$REPO_DIR/logs/overnight_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

ACCOUNTS=("$HOME/.claude" "$HOME/.claude-acc2")
ACC_IDX=0
CONSECUTIVE_LIMITS=0
MAX_TURNS=40
TASK_TIMEOUT="25m"
MAX_CYCLES=5          # Up to 5 self-planned batches (~30-40 tasks max overnight)
CYCLE=0

cd "$REPO_DIR"
git checkout -B "overnight/$(date +%Y%m%d)"

echo "=== Autonomous Orchestrator started at $(date) ===" | tee -a "$LOG_DIR/orchestrator.log"

run_claude() {
    local model="$1"
    local prompt="$2"
    local logfile="$3"
    timeout "$TASK_TIMEOUT" env CLAUDE_CONFIG_DIR="${ACCOUNTS[$ACC_IDX]}" \
        claude -p "$prompt" \
        --model "$model" \
        --max-turns "$MAX_TURNS" \
        --output-format stream-json --verbose \
        --dangerously-skip-permissions \
        > "$logfile" 2>&1
    return $?
}

handle_rate_limit() {
    local logfile="$1"
    if grep -qiE "rate limit|usage limit|overloaded|429|quota" "$logfile"; then
        CONSECUTIVE_LIMITS=$((CONSECUTIVE_LIMITS + 1))
        ACC_IDX=$(( (ACC_IDX + 1) % ${#ACCOUNTS[@]} ))
        if [[ $CONSECUTIVE_LIMITS -ge ${#ACCOUNTS[@]} ]]; then
            echo "[BOTH LIMITED] Sleeping 15m..." | tee -a "$LOG_DIR/orchestrator.log"
            CONSECUTIVE_LIMITS=0
            sleep 900
        else
            echo "[RATE LIMIT] Switching to Account $((ACC_IDX + 1))..." | tee -a "$LOG_DIR/orchestrator.log"
            sleep 15
        fi
        return 0
    fi
    CONSECUTIVE_LIMITS=0
    return 1
}

while true; do
    NEXT_TASK=$(grep -m 1 '^- \[ \]' "$TASKS_FILE" || true)

    # Queue is empty -> run Opus Self-Audit + Auto-Planner for next batch
    if [[ -z "$NEXT_TASK" ]]; then
        CYCLE=$((CYCLE + 1))
        if [[ $CYCLE -gt $MAX_CYCLES ]]; then
            echo "=== [DONE] Reached MAX_CYCLES ($MAX_CYCLES) at $(date) ===" | tee -a "$LOG_DIR/orchestrator.log"
            break
        fi

        PLAN_LOG="$LOG_DIR/$(date +%H%M%S)_opus_audit_cycle${CYCLE}.log"
        echo "[$(date +%H:%M)] [CYCLE $CYCLE] Running Opus Self-Audit & Next-Batch Planner (Acc $((ACC_IDX + 1)))..." | tee -a "$LOG_DIR/orchestrator.log"

        PLAN_PROMPT="You are the autonomous Opus Tech Lead for CountOnce.
1. Read CLAUDE.md, docs/PLAN.md, TASKS_OVERNIGHT.md, and docs/GATES.md.
2. Run 'pytest' and 'ruff check .' and inspect recent git commits to audit what was built so far.
3. Append 5-8 NEW unchecked tasks ('- [ ] ...') to TASKS_OVERNIGHT.md in priority order:
   - First: any bugs, stubs, missing tests, or PLAN.md gaps found in your audit.
   - Next: the next autonomous, non-human-gated engineering tasks from docs/PLAN.md (finish Core code/CLI/v2/baselines/benchmarks first, then Strong tier scaffolding/tests/infra).
   - STRICT RULE: Do NOT queue human-only tasks (interviews, professor meeting, filming real captures, gated HF downloads). Log any human actions needed in docs/GATES.md instead.
4. Commit TASKS_OVERNIGHT.md and docs/GATES.md.
5. If NO autonomous tasks remain anywhere in PLAN.md, do not add any '- [ ]' lines."

        run_claude "opus" "$PLAN_PROMPT" "$PLAN_LOG"
        if handle_rate_limit "$PLAN_LOG"; then
            CYCLE=$((CYCLE - 1))
            continue
        fi

        NEXT_TASK=$(grep -m 1 '^- \[ \]' "$TASKS_FILE" || true)
        if [[ -z "$NEXT_TASK" ]]; then
            echo "=== [DONE] No more autonomous tasks to queue at $(date) ===" | tee -a "$LOG_DIR/orchestrator.log"
            break
        fi
    fi

    TASK_CLEAN="${NEXT_TASK#- \[ \] }"
    SAFE_NAME=$(echo "$TASK_CLEAN" | tr -cs '[:alnum:]' '_' | cut -c1-40)
    TASK_LOG="$LOG_DIR/$(date +%H%M%S)_${SAFE_NAME}.log"

    echo "[$(date +%H:%M)] Starting (Acc $((ACC_IDX + 1))): $TASK_CLEAN" | tee -a "$LOG_DIR/orchestrator.log"

    WORK_PROMPT="You are running in unattended overnight mode.
1. Read CLAUDE.md, docs/PLAN.md, and the tail of $HANDOFF_FILE (if it exists).
2. Execute ONLY this task:
$TASK_CLEAN

3. Make the recommended engineering choice for any ambiguity; never ask questions.
4. Run 'ruff check .' and 'pytest' and fix all failures.
5. Commit your changes on the current branch (never push).
6. Mark this exact task [x] in TASKS_OVERNIGHT.md.
7. Append a brief handoff summary to $HANDOFF_FILE.
8. If truly blocked (missing credentials, gated weights, human input), log it in docs/GATES.md and exit cleanly."

    run_claude "sonnet" "$WORK_PROMPT" "$TASK_LOG"
    EXIT_CODE=$?

    if handle_rate_limit "$TASK_LOG"; then
        continue
    fi

    if grep -Fqx -- "$NEXT_TASK" "$TASKS_FILE"; then
        echo "[SKIPPED/BLOCKED] Exit $EXIT_CODE on: $TASK_CLEAN. Marking [!] and continuing." | tee -a "$LOG_DIR/orchestrator.log"
        sed -i "0,/^- \[ \] /{s/^- \[ \] /- [!] /}" "$TASKS_FILE"
        git add -A && git commit -m "wip: partial progress before roadblock on $SAFE_NAME" || true
    else
        echo "[COMPLETED] $TASK_CLEAN" | tee -a "$LOG_DIR/orchestrator.log"
    fi

    sleep 5
done
