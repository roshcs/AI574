# Security Incident: Exposed Credentials

## Summary

API keys and credentials were committed in plaintext inside
`notebooks/Multi_Domain_Agent.ipynb`. Google has already flagged at least one
key as leaked (403 PERMISSION_DENIED). All exposed credentials should be
treated as **compromised**.

## Credentials Requiring Immediate Rotation

| Provider | Credential | Action |
|----------|-----------|--------|
| Google AI (Gemini) | `GOOGLE_API_KEY` (two distinct keys) | Revoke in [Google AI Studio](https://aistudio.google.com/apikey) and generate new keys |
| Kaggle | `KAGGLE_KEY` + `KAGGLE_USERNAME` | Regenerate at [Kaggle Account Settings](https://www.kaggle.com/settings) → API → Create New Token |

## Post-Rotation Steps

1. **Do not** paste new keys into notebooks or tracked files.
2. Use one of these secure alternatives in Colab:
   - `from google.colab import userdata; os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")`
   - `os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter GOOGLE_API_KEY: ")`
3. For local development, use a `.env` file (already in `.gitignore`).
4. Consider scrubbing git history with `git filter-repo` or BFG Repo Cleaner
   if this repository was ever public.

## Preventive Controls Added

- **Pre-commit hook**: `nbstripout` strips notebook outputs before commit.
- **CI secret scanning**: GitHub workflow step blocks sync if API-key patterns
  are found in tracked files.
- Notebook cells now default to `userdata.get(...)` with `getpass` fallback.
