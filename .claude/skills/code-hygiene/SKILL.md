---
name: code-hygiene
description: Audit code for unnecessary comments, naming quality, and clean structure. Use when the user asks to review code cleanliness/readability/style, check comments/naming/structure, or tidy up before a PR or before merging dev into main. NOT for finding duplicate/overlapping functions — that's the overlap-detector subagent — and not for bug hunting.
---

# Code Hygiene

Audit code for three things only: **unnecessary comments**, **naming quality**, and **clean
structure**. This is a read-and-report skill — do not change code unless the user explicitly
asks for fixes. Finding functions that do the same thing is out of scope; defer that to the
`overlap-detector` subagent.

## Scope

Pick the narrowest scope that fits the request:

1. Files the user names, if any.
2. Otherwise, changed files vs. the base branch:
   `git diff --name-only main...HEAD` (fall back to `git diff --name-only` for uncommitted work).
3. If neither applies, the whole source tree under `stock-agent/src`.

**Always exclude** `venv/`, `__pycache__/`, generated files, and anything in `.gitignore`.
Only audit source the user actually authored.

## What to flag

### 1. Unnecessary comments
The test is "what vs why." **Keep** comments that explain *why* — non-obvious intent,
trade-offs, gotchas, leakage/look-ahead reasoning, units. This repo has good ones (e.g. the
no-look-ahead notes in `pipeline.py`, the atomic-write rationale in `news/sentiment.py`) —
do not flag those.

**Flag** comments that add no information:
- Restating the code: `i += 1  # increment i`, `# loop over tickers` above an obvious loop.
- Docstrings that only echo the signature ("Add label. Adds the label.").
- Commented-out / dead code left behind.
- Stale comments that no longer match the code.
- Section-divider noise and redundant headers.

### 2. Naming
- **Non-descriptive names**: `tmp`, `data`, `df2`, `x`, `res`, `val`, `thing` where a domain
  term exists. Single letters only for loop indices, short math, or `df`/`X`/`y` conventions.
- **Inconsistent conventions**: this codebase is `snake_case` for functions/vars,
  `UPPER_SNAKE` for module constants. Flag `camelCase`/`PascalCase` locals, or the same
  concept named differently across files (`ticker` vs `symbol` vs `sym`, `exc` vs `e`).
- **Misleading names**: name implies one thing, value/return is another (a `list` named
  `*_map`; a `get_*` that mutates; a generic `_load` that only loads close prices).
- **Abbreviation / redundancy**: `cfg` here vs `config` there; over-qualified names.

### 3. Clean structure
- Functions doing too much / too long — suggest a split along its natural seams.
- Deep nesting that an early `return`/guard clause would flatten.
- Dead code, unused imports/params, unreachable branches, no-op statements.
- Inconsistent module layout (helpers above vs below callers), or logic that belongs in a
  different module.
- Magic numbers that should be named constants (note: many already live in `config.py`).

## How to report

Group findings by file, ordered by impact (structure, then naming, then comments). For each:

```
<file>:<line>  <short label>
  Issue: <one line>
  Suggest: <concrete fix>
```

End with a 1–2 line summary (counts + the single highest-value cleanup). Then ask whether the
user wants the fixes applied. Do not edit until they say yes.

## Notes
- Overlapping/duplicate functions are NOT handled here — recommend running the
  `overlap-detector` subagent for that.
- Overlaps with `/code-review` and `/simplify` but is intentionally narrow (comments + naming
  + structure). Don't expand into bug hunting.
- Runs on demand. For "always run on every change," that's a hook, not a skill — offer to set
  one up if the user wants that.
