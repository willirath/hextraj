# Agent guidelines for hextraj

## Repository rules

- **No pushing or committing** unless explicitly asked.
- **Discuss before committing** — present what you did and any findings, wait for the go-ahead.
- Plans go in `dev/plans/*.md`.
- Agent-facing documentation goes in `dev/docs/*.md`.

## Environment

- Always use `pixi run <command>` — never bare `python`, `pytest`, `jupyter`, etc.
- Examples: `pixi run python`, `pixi run pytest`, `pixi run jupyter nbconvert ...`

## TDD workflow

All non-trivial implementation work follows a test-first loop using subagents:

1. **Test agent (haiku or similar)** — writes failing tests only; does not implement.
2. **Implementation agent (haiku or similar)** — makes the tests pass; does not add tests.
3. **Main agent** — runs the tests, reviews results, escalates to a more powerful agent if lower agent struggles, then reports to the user before committing.

Rules:
- Main agent does not write implementation code directly — delegate to subagents.
- Prefer haiku or other simpler model for both test and implementation agents; escalate to higher model only if the task is complex or simpler agent fails.
- Test style: plain `pytest` functions with `@pytest.mark.parametrize`; no test classes.
- **Every agent must read `AGENTS.md` at the start of its task.**

## Backwards compatibility

This project does not maintain backwards compatibility. Remove dead code without ceremony — no deprecation warnings, no deprecation wrappers, no `_removed` stubs, no tests asserting that something was deleted. If something is gone, it's gone.

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

**Editing notebooks:** Use the `NotebookEdit` tool to add, modify, or delete cells — never manipulate `.ipynb` JSON directly with Python scripts.
