# Verification Command

Run comprehensive verification on current codebase state.

## Instructions

Execute verification in this exact order. A check whose tool is not installed is reported as not run, with the reason; never as passed, and do not install tools for it (CLAUDE.md §〇.8).

1. **Type Check**
   - No type-check command is required (CLAUDE.md §〇.8). If mypy is installed, run mypy src scripts and report the result
     as informational (import-untyped errors only mean the pandas / geopandas stubs are missing); otherwise report as not run

2. **Lint Check**
   - Run ruff check src scripts tests (rules in pyproject.toml)
   - Report warnings and errors

3. **Test Suite**
   - Run python -m pytest
   - Report pass/fail count
   - No coverage threshold is configured; report coverage only if pytest-cov is already installed

4. **Security Check**
   - Run pip-audit if it is installed
   - Check for hardcoded secrets (grep -r "sk-" etc.) and for files .claude/rules/security.md forbids committing

5. **Print Audit**
   - Search for print() in src/ (the CLI's output in cli.py is intended)
   - Report locations

6. **Git Status**
   - Show uncommitted changes
   - Show files modified since last commit

## Output

Produce a concise verification report:

```
VERIFICATION: [PASS/FAIL]

Types:    [OK/X errors (informational)/not run]
Lint:     [OK/X issues]
Tests:    [X/Y passed]
Security: [OK/X vulnerabilities/not run]
Secrets:  [OK/X found]
Prints:   [OK/X print() statements]

Ready for commit: [YES/NO]
```

If any critical issues, list them with fix suggestions.

## Arguments

$ARGUMENTS can be:
- `quick` - Only types + lint
- `full` - All checks (default)
- `pre-commit` - Checks relevant for commits
- `pre-pr` - Full checks plus security scan
