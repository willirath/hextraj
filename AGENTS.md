# Agent guidelines for hextraj

## Repository rules

- **No pushing or committing** unless explicitly asked.
- Plans go in `dev/plans/*.md`.
- Agent-facing documentation goes in `dev/docs/*.md`.

## Python style

Be Pythonic. Don't over-engineer:

- No excessive type annotations or runtime type checks.
- No defensive error handling for misuse of user-facing functions — let them raise naturally so the human sees the real error.
- Prefer simple, idiomatic constructs over clever abstractions.

## Jupyter notebooks

Notebooks follow a **human-facing literate programming** style:

- Use Markdown cells for narrative text, equations (LaTeX), and explanations — not inline comments inside code cells.
- Keep code cells well-scoped and short: one logical step per cell.
- Avoid long monolithic code blocks; split them into meaningful, readable units.
