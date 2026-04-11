#!/usr/bin/env bash
# batch_analyze.sh — Automated batch paper analysis via Claude Code CLI
#
# Each session processes 8 papers (2 batches × 4 PDFs).
# After 8 papers, the session ends and a new one starts automatically.
# Stops when no more "Downloaded" entries remain in analysis_log.csv.
#
# Usage:
#   cd <repo_root>          # the folder containing paperPDFs/ and paperAnalysis/
#   bash scripts/batch_analyze.sh [--dry-run] [--max-sessions N]
#
# Requirements:
#   - Claude Code CLI (`claude`) installed and authenticated
#   - analysis_log.csv with "Downloaded" entries to process

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
PAPERS_PER_SESSION=8          # 2 batches × 4 PDFs
LOG_FILE="paperAnalysis/analysis_log.csv"
SKILL_PROMPT='Run /papers-analyze-pdf on the next batch of Downloaded entries in analysis_log.csv. Process 4 PDFs per batch, 2 batches total (8 PDFs). After finishing, output the list of written files and any analysis_mismatch entries.'
MAX_SESSIONS="${2:-999}"      # default: unlimited
DRY_RUN=false
SESSION_COUNT=0
COOLDOWN=120                   # seconds between sessions

# ── Parse args ──────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=true ;;
    --max-sessions) shift ;;   # value captured by ${2:-999} above
    --max-sessions=*) MAX_SESSIONS="${arg#*=}" ;;
  esac
done

# ── Functions ───────────────────────────────────────────────────────
count_remaining() {
  grep -c ',Downloaded,' "$LOG_FILE" 2>/dev/null || echo 0
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

# ── Pre-flight checks ──────────────────────────────────────────────
if [[ ! -f "$LOG_FILE" ]]; then
  echo "ERROR: $LOG_FILE not found. Run from repo root." >&2
  exit 1
fi

if ! command -v claude &>/dev/null; then
  echo "ERROR: claude CLI not found. Install: https://docs.anthropic.com/en/docs/claude-code" >&2
  exit 1
fi

REMAINING=$(count_remaining)
log "Starting batch analysis. Downloaded entries remaining: $REMAINING"

if [[ "$REMAINING" -eq 0 ]]; then
  log "No Downloaded entries to process. Done."
  exit 0
fi

# ── Main loop ───────────────────────────────────────────────────────
while true; do
  REMAINING=$(count_remaining)

  if [[ "$REMAINING" -eq 0 ]]; then
    log "All entries processed. Total sessions: $SESSION_COUNT"
    break
  fi

  if [[ "$SESSION_COUNT" -ge "$MAX_SESSIONS" ]]; then
    log "Reached max sessions ($MAX_SESSIONS). Remaining: $REMAINING"
    break
  fi

  SESSION_COUNT=$((SESSION_COUNT + 1))
  BATCH_SIZE=$((REMAINING < PAPERS_PER_SESSION ? REMAINING : PAPERS_PER_SESSION))

  log "━━━ Session $SESSION_COUNT ━━━ Processing $BATCH_SIZE papers ($REMAINING remaining)"

  if [[ "$DRY_RUN" == true ]]; then
    log "[DRY RUN] Would run: claude -p \"$SKILL_PROMPT\""
    # Simulate progress for dry run
    sleep 1
  else
    # Run Claude Code CLI in non-interactive mode
    # --no-input: don't wait for user input
    # The prompt tells Claude to process 2 batches of 4
    claude -p "$SKILL_PROMPT" \
      --allowedTools "Read,Write,Shell,Glob,Grep" \
      2>&1 | tee -a "paperAnalysis/batch_analyze_session_${SESSION_COUNT}.log"

    log "Session $SESSION_COUNT complete."
  fi

  # Check if progress was made (avoid infinite loop on stuck entries)
  NEW_REMAINING=$(count_remaining)
  if [[ "$NEW_REMAINING" -eq "$REMAINING" ]]; then
    log "WARNING: No progress in session $SESSION_COUNT ($REMAINING still remaining)."
    log "Possible causes: all remaining entries are stuck (too_large, analysis_mismatch, or PDF missing)."
    log "Check the session log: paperAnalysis/batch_analyze_session_${SESSION_COUNT}.log"
    break
  fi

  if [[ "$NEW_REMAINING" -gt 0 ]]; then
    log "Cooling down ${COOLDOWN}s before next session..."
    sleep "$COOLDOWN"
  fi
done

log "━━━ Batch analysis finished ━━━"
log "Sessions run: $SESSION_COUNT"
log "Remaining Downloaded: $(count_remaining)"
log "Check paperAnalysis/batch_analyze_session_*.log for details."
