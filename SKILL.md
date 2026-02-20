---
name: claude-reflect
description: Self-learning system that captures corrections during sessions and reminds users to run /reflect to update CLAUDE.md. Use when discussing learnings, corrections, or when the user mentions remembering something for future sessions.
---

# Claude Reflect - Self-Learning System

A two-stage system that helps Claude Code learn from user corrections.

## How It Works

**Stage 1: Capture (Automatic)**
Hooks detect `remember:` markers and pass other messages through a structural false-positive filter, queuing candidates to `~/.claude/learnings-queue.json`.

**Stage 2: Process (Manual)**
User runs `/reflect` to review and apply queued learnings to CLAUDE.md files.

## Available Commands

| Command | Purpose |
|---------|---------|
| `/reflect` | Process queued learnings with human review |
| `/reflect --scan-history` | Scan past sessions for missed learnings |
| `/reflect --dry-run` | Preview changes without applying |
| `/reflect-skills` | Discover skill candidates from repeating patterns |
| `/skip-reflect` | Discard all queued learnings |
| `/view-queue` | View pending learnings without processing |

## When to Remind Users

Remind users about `/reflect` when:
- They complete a feature or meaningful work unit
- They make corrections you should remember for future sessions
- They explicitly say "remember this" or similar
- Context is about to compact and queue has items

## Correction Detection

Uses a two-stage detection approach:

**Real-time (automatic, <1ms):**
1. **Regex: "remember:"** — Explicit marker, highest priority, never misses
2. **Regex: False positive filter** — Structural patterns (questions, task requests), language-agnostic
3. **Passthrough** — Everything else is queued as "auto" with low confidence

**During /reflect (Claude AI):**
- Semantic validation filters false positives with high accuracy
- Extracts concise, actionable statements
- Works for any language

Also detects:
- Tool rejections (user stops an action with guidance)
- Positive feedback ("perfect!", "great approach")

## Learning Destinations

- `~/.claude/CLAUDE.md` - Global learnings (model names, general patterns)
- `./CLAUDE.md` - Project-specific learnings (conventions, tools, structure)
- `./CLAUDE.local.md` - Personal learnings (machine-specific, gitignored)
- `./.claude/rules/*.md` - Modular rules with optional path-scoping
- `~/.claude/rules/*.md` - Global modular rules
- `~/.claude/projects/<project>/memory/*.md` - Auto memory (low-confidence, exploratory)
- `commands/*.md` - Skill improvements (corrections during skill execution)

## Example Interaction

```
User: no, use gpt-5.1 not gpt-5 for reasoning tasks
Claude: Got it, I'll use gpt-5.1 for reasoning tasks.

[Hook captures this correction to queue]

User: /reflect
Claude: Found 1 learning queued. "Use gpt-5.1 for reasoning tasks"
        Scope: global
        Apply to ~/.claude/CLAUDE.md? [y/n]
```
