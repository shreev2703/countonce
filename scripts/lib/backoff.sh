#!/usr/bin/env bash
# Error-code parsing and exponential backoff for long-running scripts. Source this file.
#
#   classify_limit LOGFILE          -> quota | rate | overloaded | none
#   retry_after_secs LOGFILE        -> seconds the provider asked us to wait (empty if it did not)
#   backoff_seconds KIND N [HINT]   -> seconds to wait before retry N (1 = first)
#
# Only the tail of the log is inspected: a stream-json transcript can mention "429" or
# "rate limit" in ordinary output, but a real failure is the last thing the CLI prints.

BACKOFF_TAIL_LINES="${BACKOFF_TAIL_LINES:-20}"
BACKOFF_CAP="${BACKOFF_CAP:-1800}"        # never wait longer than this (seconds)
BACKOFF_JITTER="${BACKOFF_JITTER:-10}"    # up to this percent extra; 0 disables (tests)

_log_tail() { tail -n "$BACKOFF_TAIL_LINES" "$1" 2>/dev/null | tr '[:upper:]' '[:lower:]'; }

classify_limit() {
    local t
    t=$(_log_tail "$1")
    if grep -qE 'usage limit|insufficient_quota|exceeded your current quota|credit balance is too low|quota exceeded|hit your limit' <<<"$t"; then
        echo quota
    elif grep -qE '"status"? ?[:=] ?429|http 429|status(_code)? ?[:=] ?429|rate[ _-]?limit|too many requests' <<<"$t"; then
        echo rate
    elif grep -qE 'overloaded|"status"? ?[:=] ?(529|503)|http (529|503)' <<<"$t"; then
        echo overloaded
    else
        echo none
    fi
}

retry_after_secs() {
    local t n unit reset now
    t=$(_log_tail "$1")
    # "usage limit reached|<unix epoch of reset>" (Claude Code CLI)
    reset=$(grep -oE 'limit reached\|[0-9]{9,}' <<<"$t" | tail -n 1 | grep -oE '[0-9]+$')
    if [[ -n "$reset" ]]; then
        now=$(date +%s)
        (( reset > now )) && echo $(( reset - now + 30 )) && return 0
    fi
    # "retry-after: 42" / "retry after 42" / "retry in 42s"
    n=$(grep -oE 'retry[ _-]?(after|in)"?[: =]+[0-9]+' <<<"$t" | tail -n 1 | grep -oE '[0-9]+$')
    if [[ -n "$n" ]]; then echo "$n"; return 0; fi
    # "try again in 5 minutes" / "try again in 30 seconds"
    read -r n unit < <(grep -oE 'try again in [0-9]+ ?(second|sec|minute|min|s|m)' <<<"$t" |
        tail -n 1 | sed -E 's/try again in ([0-9]+) ?([a-z]+)/\1 \2/')
    if [[ -n "${n:-}" ]]; then
        [[ "$unit" == m* ]] && n=$(( n * 60 ))
        echo "$n"
    fi
    return 0
}

backoff_seconds() {
    local kind="$1" attempt="$2" hint="${3:-}" base delay extra
    case "$kind" in
        quota) base=300 ;;       # credit does not return in a minute
        overloaded) base=60 ;;
        *) base=30 ;;
    esac
    (( attempt < 1 )) && attempt=1
    if [[ -n "$hint" ]]; then
        delay=$hint
    else
        delay=$(( base << (attempt > 20 ? 20 : attempt - 1) ))
    fi
    extra=$(( BACKOFF_JITTER > 0 ? RANDOM % (delay * BACKOFF_JITTER / 100 + 1) : 0 ))
    delay=$(( delay + extra ))
    (( delay > BACKOFF_CAP )) && delay=$BACKOFF_CAP
    echo "$delay"
}
