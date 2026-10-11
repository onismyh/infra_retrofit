# Security Rules

## Secrets Management

### Never Store Secrets in Git-Tracked Files

- API keys, tokens, passwords must NEVER appear in committed files
- Use environment variables or `.env` files (which are gitignored)
- `.claude/settings.json` / `.claude/settings.local.json` contain sensitive tokens and are excluded from Git via `.gitignore`
  (`.vscode/settings.json` holds only editor settings and is tracked)

### Environment Variables

```python
import os

# GOOD: Read from environment
api_key = os.environ["API_KEY"]

# BAD: Hardcoded secret
api_key = "sk-abc123..."
```

### `.env` File Usage

```bash
# .env (gitignored)
ANTHROPIC_AUTH_TOKEN=sk-ant-...
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
WANDB_API_KEY=...
```

```python
# Load in code
from dotenv import load_dotenv
load_dotenv()
```

## Sensitive File Warnings

The following files must NEVER be committed to Git:

| File Pattern | Reason |
|-------------|--------|
| `.claude/settings.json`, `.claude/settings.local.json` | Contain API tokens |
| `.env`, `.env.*` | Environment secrets |
| `*.pem`, `*.key` | Private keys |
| `credentials.json` | Service account credentials |
| `*_secret*`, `*_token*` | Named secret files |
| `*.sqlite`, `*.db` | May contain user data |
| `gurobi.lic` | Gurobi licence file (see AGENTS.md) |

## Code Security

### Prohibited in Source Code

- Hardcoded passwords or API keys
- Hardcoded IP addresses or internal URLs (use config)
- Disabled SSL verification without justification
- `eval()` or `exec()` with user input
- SQL string concatenation (use parameterized queries)

### Pre-Commit Checks

The `security-guard.js` hook automatically checks for:
- Secrets in file content before write/edit operations
- Dangerous bash commands
- Sensitive file modifications

## Token Rotation

If a token is accidentally committed:
1. Immediately rotate the compromised token
2. Use `git filter-branch` or BFG Repo-Cleaner to remove from history
3. Force push the cleaned history
4. Verify the old token is invalidated
