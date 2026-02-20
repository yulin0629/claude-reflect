# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

claude-reflect is a Claude Code plugin that implements a two-stage self-learning system:
1. **Capture Stage** (automatic): Hooks detect correction patterns in user prompts and queue them
2. **Process Stage** (manual): `/reflect` command processes queued learnings with human review and writes to CLAUDE.md files

## Architecture

```
.claude-plugin/plugin.json  → Plugin manifest, points to hooks
hooks/hooks.json            → Hook definitions (SessionStart, UserPromptSubmit, PreCompact, PostToolUse)
scripts/                    → Python scripts for hooks and extraction
scripts/lib/                → Shared utilities (reflect_utils.py, semantic_detector.py)
scripts/legacy/             → Deprecated bash scripts (for reference)
commands/*.md               → Skill definitions for /reflect, /reflect-skills, /skip-reflect, /view-queue
SKILL.md                    → Context provided when plugin is invoked
tests/                      → Test suite (pytest)
```

### Data Flow

1. User prompt → `capture_learning.py` (UserPromptSubmit hook) → `~/.claude/learnings-queue.json`
2. `/reflect` command → reads queue + scans sessions → filters/dedupes → routes to memory targets
3. Session files live at `~/.claude/projects/[PROJECT_FOLDER]/*.jsonl`

### Memory Targets (Full Hierarchy)

| Target | Path | Type | Description |
|--------|------|------|-------------|
| Global CLAUDE.md | `~/.claude/CLAUDE.md` | `global` | Always enabled |
| Project CLAUDE.md | `./CLAUDE.md` | `root` | Project-specific |
| CLAUDE.local.md | `./CLAUDE.local.md` | `local` | Personal, gitignored |
| Subdirectory | `./**/CLAUDE.md` | `subdirectory` | Auto-discovered |
| Project Rules | `./.claude/rules/*.md` | `rule` | Modular, path-scoped |
| User Rules | `~/.claude/rules/*.md` | `user-rule` | Global modular rules |
| Auto Memory | `~/.claude/projects/<project>/memory/*.md` | `auto-memory` | Low-confidence staging |
| Skill Files | `./commands/*.md` | skill | Correction during skill use |
| AGENTS.md | `./AGENTS.md` | agents | Cross-tool standard |

### Key Files

- `scripts/lib/reflect_utils.py`: Shared utilities (paths, queue ops, detect_patterns, memory hierarchy discovery, auto memory, rule frontmatter parsing)
- `scripts/lib/semantic_detector.py`: AI-powered semantic analysis via `claude -p`
- `scripts/capture_learning.py`: Pattern detection (explicit markers, passthrough) with confidence scoring
- `scripts/check_learnings.py`: PreCompact hook that backs up queue before context compaction
- `scripts/extract_session_learnings.py`: Extracts user messages from session JSONL files
- `scripts/extract_tool_rejections.py`: Extracts user corrections from tool rejections
- `scripts/compare_detection.py`: Compare regex vs semantic detection on session data
- `commands/reflect.md`: Main skill defining the /reflect workflow (memory hierarchy aware)
- `commands/reflect-skills.md`: Skill discovery - AI-powered pattern detection from sessions

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

# Test passthrough capture (queued as auto, validated by /reflect AI)
echo '{"prompt":"不要用那個，改用這個"}' | python3 scripts/capture_learning.py
```

## Plugin Structure

The plugin registers via `.claude-plugin/plugin.json`:
- Hooks are defined in `hooks/hooks.json`
- Commands (skills) are markdown files in `commands/`
- `SKILL.md` provides context when the plugin is active

### Hook Events

| Hook | Script | Purpose |
|------|--------|---------|
| SessionStart | `session_start_reminder.py` | Show pending learnings reminder |
| UserPromptSubmit | `capture_learning.py` | Detect corrections and queue them |
| PreCompact | `check_learnings.py` | Backup queue before compaction |
| PostToolUse (Bash) | `post_commit_reminder.py` | Remind to /reflect after commits |

## Detection Methods

### Two-Stage Detection

**Stage 1: Real-time Capture** (`detect_patterns()` in `scripts/lib/reflect_utils.py`, <1ms)

1. **Regex: "remember:"** — Explicit marker, highest priority, never misses
2. **Regex: Guardrail patterns** — "don't do X unless" constraints, high confidence (0.85-0.90)
3. **Regex: False positive filter** — Structural patterns (questions, task requests, error descriptions), language-agnostic, includes CJK question particles
4. **Regex: Positive/Correction patterns** — English-centric patterns with confidence scoring (0.55-0.90)
5. **Passthrough** — Non-English or unmatched messages queued as "auto" (0.50) for /reflect AI validation

**Stage 2: AI Validation** (during `/reflect`)

`scripts/lib/semantic_detector.py` provides AI-powered validation:
- Uses `claude -p --output-format json` for semantic analysis
- **Multi-language support** — works for any language, not just English
- **High accuracy** — filters out false positives
- **Cleaner learnings** — extracts concise, actionable statements

Key functions:
- `semantic_analyze(text)` — analyze single message
- `validate_queue_items(items)` — batch validate queue items

## Session File Format

Session files are JSONL at `~/.claude/projects/[PROJECT_FOLDER]/`:
- User messages: `{"type": "user", "message": {"content": [{"type": "text", "text": "..."}]}, "isMeta": false}`
- Tool rejections: `{"type": "user", "message": {"content": [{"type": "tool_result", "is_error": true, "content": "...the user said:\n[feedback]"}]}}`
- Filter `isMeta: true` to exclude command expansions

## Queue Item Structure

```json
{
  "type": "auto|explicit|positive|guardrail",
  "message": "user's original text",
  "timestamp": "ISO8601",
  "project": "/path/to/project",
  "patterns": "matched pattern names",
  "confidence": 0.75,
  "sentiment": "correction|positive",
  "decay_days": 90
}
```

## Skill Discovery (/reflect-skills)

Analyzes session history to discover repeating patterns that could become skills.

**Design Principles:**
- **AI-powered** — Claude uses reasoning to identify patterns, not regex
- **Semantic similarity** — detects same intent across different phrasings
- **Human-in-the-loop** — user approves before skill generation

**Usage:**
```bash
/reflect-skills              # Analyze last 14 days
/reflect-skills --days 30    # Analyze last 30 days
/reflect-skills --dry-run    # Preview without generating files
```

**What it detects:**
- Workflow patterns (repeated multi-step sequences)
- Misunderstanding patterns (corrections that could become guardrails)
- Intent similarity (same goal, different wording)

## Skill Improvement Routing

When running `/reflect`, corrections made during skill execution can be routed back to the skill file itself.

**How it works:**
1. `/reflect` detects when a correction followed a skill invocation (e.g., `/deploy`)
2. Claude reasons about whether the correction relates to the skill's workflow
3. User is offered routing options: skill file | CLAUDE.md | both
4. Skill file is updated in the appropriate section (steps, guardrails, etc.)

**Example:**
```
User: /deploy
Claude: [deploys without running tests]
User: "no, always run tests before deploying"

→ /reflect detects this relates to /deploy
→ Offers to add "Run tests before deploying" to commands/deploy.md
→ Skill file updated with new step in workflow
```

## Platform Support

- **macOS**: Fully supported
- **Linux**: Fully supported
- **Windows**: Fully supported (native Python, no WSL required)

Requires Python 3.6+.

## Releasing

See [RELEASING.md](RELEASING.md) for version bump checklist and release process.
