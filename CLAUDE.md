# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

claude-reflect is a Claude Code plugin that implements a two-stage self-learning system:
1. **Capture Stage** (automatic): Hooks detect correction patterns in user prompts and queue them
2. **Process Stage** (manual): `/reflect` command processes queued learnings with human review and writes to CLAUDE.md files

## Architecture

```
.claude-plugin/plugin.json  → Plugin manifest, points to hooks
hooks/hooks.json            → Hook definitions (PreCompact, PostToolUse)
scripts/                    → Python scripts for hooks and extraction
scripts/lib/                → Shared utilities (reflect_utils.py)
scripts/legacy/             → Deprecated bash scripts (for reference)
commands/*.md               → Skill definitions for /reflect, /skip-reflect, /view-queue
SKILL.md                    → Context provided when plugin is invoked
tests/                      → Test suite (pytest)
```

### Data Flow

1. User prompt → `capture_learning.py` (UserPromptSubmit hook) → `~/.claude/learnings-queue.json`
2. `/reflect` command → reads queue + scans sessions → filters/dedupes → writes to CLAUDE.md/AGENTS.md
3. Session files live at `~/.claude/projects/[PROJECT_FOLDER]/*.jsonl`

### Key Files

- `scripts/lib/reflect_utils.py`: Shared utilities (paths, queue ops, pattern detection)
- `scripts/capture_learning.py`: Pattern detection (correction, positive, explicit markers) with confidence scoring
- `scripts/check_learnings.py`: PreCompact hook that backs up queue before context compaction
- `scripts/extract_session_learnings.py`: Extracts user messages from session JSONL files
- `scripts/extract_tool_rejections.py`: Extracts user corrections from tool rejections
- `commands/reflect.md`: Main skill - 800+ line document defining the /reflect workflow

## Development Commands

```bash
# Test capture hook with simulated input
echo '{"prompt":"no, use gpt-5.1 not gpt-5"}' | python3 scripts/capture_learning.py

# View current learnings queue
cat ~/.claude/learnings-queue.json

# Test session extraction
python3 scripts/extract_session_learnings.py ~/.claude/projects/[PROJECT]/*.jsonl --corrections-only

# Run tests
python -m pytest tests/ -v

# Clear queue for testing
echo "[]" > ~/.claude/learnings-queue.json
```

## Plugin Structure

The plugin registers via `.claude-plugin/plugin.json`:
- Hooks are defined in `hooks/hooks.json`
- Commands (skills) are markdown files in `commands/`
- `SKILL.md` provides context when the plugin is active

### Hook Events

| Hook | Script | Purpose |
|------|--------|---------|
| PreCompact | `check_learnings.py` | Backup queue before compaction |
| PostToolUse (Bash) | `post_commit_reminder.py` | Remind to /reflect after commits |

Note: UserPromptSubmit hook for `capture_learning.py` is configured by the user per Claude Code plugin system.

## Pattern Detection

`scripts/lib/reflect_utils.py` defines pattern detection:
- **Corrections**: "no, use X", "don't use", "stop using", "that's wrong", "actually", "use X not Y"
- **Positive**: "perfect!", "exactly right", "great approach", "nailed it"
- **Explicit**: "remember:" prefix (highest confidence)

Confidence scores range 0.60-0.90 based on pattern strength and count.

## Session File Format

Session files are JSONL at `~/.claude/projects/[PROJECT_FOLDER]/`:
- User messages: `{"type": "user", "message": {"content": [{"type": "text", "text": "..."}]}, "isMeta": false}`
- Tool rejections: `{"type": "user", "message": {"content": [{"type": "tool_result", "is_error": true, "content": "...the user said:\n[feedback]"}]}}`
- Filter `isMeta: true` to exclude command expansions

## Queue Item Structure

```json
{
  "type": "auto|explicit|positive",
  "message": "user's original text",
  "timestamp": "ISO8601",
  "project": "/path/to/project",
  "patterns": "matched pattern names",
  "confidence": 0.75,
  "sentiment": "correction|positive",
  "decay_days": 90
}
```

## Platform Support

- **macOS**: Fully supported
- **Linux**: Fully supported
- **Windows**: Fully supported (native Python, no WSL required)

Requires Python 3.6+.
