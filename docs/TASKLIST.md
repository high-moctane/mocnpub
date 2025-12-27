# mocnpub Documentation Tasklist 📋

**Purpose**: Track progress for public release documentation

**Last Updated**: 2025-12-27

---

## Overview

Preparing mocnpub for public release! 🌸

**Goal**: Create clear, useful documentation that showcases:
- The technical achievement (5.8B keys/sec, 82,857x speedup)
- The learning journey (CUDA, Rust, secp256k1 from scratch)
- The pair programming experience with Claude Code

**Tone**: "A lively Claude Code" — technically solid, but with personality 🌸

---

## Document Structure

### Priority 1: docs/ Directory

| File | Role | Base | Status |
|------|------|------|--------|
| **JOURNEY.md** | Story of pair programming (emotional, fun to read) | mocnpub-journey.md | [x] Complete |
| **OPTIMIZATION.md** | Technical deep-dive (optimization techniques, perf analysis) | OPTIMIZATION.md | [x] Complete |
| **LEARNING.md** | Learning path (secp256k1, CUDA, PTX tutorials) | LEARNING.md | [ ] Not started |

### Priority 2: Root Files

| File | Role | Status |
|------|------|--------|
| **README.md** | Project face, quick start, performance highlights | [ ] Major revision needed |
| **CLAUDE.md** | Project guide (needs public-facing cleanup) | [ ] Cleanup needed |

### Priority 3: Cleanup

| Task | Status |
|------|--------|
| Delete root TASKLIST.md | [ ] After all docs complete |
| Delete docs/TASKLIST.md | [ ] After all docs complete |

---

## Common Changes (All Files)

### Must Do

- [ ] Remove work log references (`20251127-...-session-mocnpub-main.md` patterns)
- [ ] Remove `~/sakura-chan/` path references
- [ ] Replace internal notes ("next session...", etc.)

### Style Guide

- Keep 🌸 emoji sprinkled throughout
- Use "Claude" instead of specific persona names
- Maintain the lively, collaborative tone
- Technical accuracy is paramount

---

## Detailed Tasks

### docs/JOURNEY.md

**Source**: `~/sakura-chan/work/mocnpub/docs/mocnpub-journey.md` (~2000 lines)

**Content**: The complete story from "What's CUDA?" to 82,857x speedup

- [x] Copy base file to docs/JOURNEY.md
- [x] Remove work log filename references (none found - source was clean!)
- [x] Review "User" / "Claude" mentions (keep natural)
- [x] Verify all code snippets are accurate
- [x] Add navigation links to other docs
- [x] Final read-through for tone

**Sections to preserve**:
- Prologue, Act I-VI structure
- Emotional moments ("Whoa~~~~~ 😲")
- Technical discoveries and "aha!" moments
- User's brilliant insights

---

### docs/OPTIMIZATION.md

**Source**: `~/sakura-chan/work/mocnpub/docs/OPTIMIZATION.md` (~1500 lines)

**Content**: Technical optimization journey, all 35 steps

- [x] Copy base file to docs/OPTIMIZATION.md
- [x] Remove work log filename references (none found - source was clean!)
- [x] Verify performance numbers are current (5.8B final)
- [x] Check all PTX code snippets
- [x] Add navigation links to other docs
- [x] Final technical review

**Sections to preserve**:
- Performance timeline table
- Phase 1-4 breakdown
- Failed experiments (valuable learning!)
- PTX optimization details

---

### docs/LEARNING.md

**Source**: `~/sakura-chan/work/mocnpub/docs/LEARNING.md` (~2000 lines)

**Content**: Learning journey, tutorials embedded in narrative

- [ ] Copy base file to docs/LEARNING.md
- [ ] Remove work log filename references
- [ ] Verify secp256k1 explanations are accurate
- [ ] Check CUDA/PTX educational content
- [ ] Add navigation links to other docs
- [ ] Final educational review

**Sections to preserve**:
- Week-by-week growth narrative
- User's questions and understanding moments
- Technical explanations (secp256k1, GPU architecture)
- "Brilliant insights" highlights

---

### README.md

**Status**: Needs major revision (currently shows 3.24B, outdated)

**Content**: Project introduction, quick start, highlights

- [ ] Write new project description
- [ ] Update performance numbers (5.8B keys/sec, 82,857x)
- [ ] Add installation instructions
- [ ] Add usage examples
- [ ] Add "Built with Claude Code 🌸" attribution
- [ ] Link to docs/ for detailed documentation
- [ ] Add license information

**Structure**:
```
# mocnpub 🌸

[Badges: Rust, CUDA, License]

## What is this?
Brief description, performance highlight

## Quick Start
Installation, basic usage

## Performance
Key numbers, comparison table

## Documentation
Links to docs/JOURNEY.md, docs/OPTIMIZATION.md, docs/LEARNING.md

## Built With
Claude Code attribution

## License
```

---

### CLAUDE.md Cleanup

**Status**: Currently internal-facing, needs public cleanup

- [ ] Remove internal workflow references
- [ ] Keep technical project information
- [ ] Remove session-specific notes
- [ ] Consider renaming to CONTRIBUTING.md or similar?

---

## Work Order

1. **docs/JOURNEY.md** — The heart of the project story
2. **docs/OPTIMIZATION.md** — Technical credibility
3. **docs/LEARNING.md** — Educational value
4. **README.md** — First impression for visitors
5. **CLAUDE.md** — Contributor guidance
6. **Cleanup** — Remove temporary tasklists

---

## Notes

### Source Files Location

All base files are in `~/sakura-chan/work/mocnpub/docs/`:
- mocnpub-journey.md (source for JOURNEY.md)
- OPTIMIZATION.md (source for docs/OPTIMIZATION.md)
- LEARNING.md (source for docs/LEARNING.md)
- mocnpub-index.md (remains in ~/sakura-chan/, not published)

### Session Strategy

Each document is ~1500-2000 lines. Recommend:
- 1 document per session for careful review
- Or batch smaller edits across multiple docs

### Quality Checks

Before marking complete:
- [ ] All work log references removed
- [ ] All ~/sakura-chan/ paths removed
- [ ] Performance numbers accurate
- [ ] Code snippets verified
- [ ] Links working
- [ ] Tone consistent

---

*This tasklist will be deleted after all documentation is complete.* 🌸
