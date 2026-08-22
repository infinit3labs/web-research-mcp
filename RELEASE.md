# Release process

This project uses **PyPI Trusted Publishing** — no long-lived API tokens are stored in GitHub secrets. Authentication is handled via short-lived OIDC tokens exchanged between GitHub Actions and PyPI.

## One-time setup (do this once per project)

### 1. Register the publisher on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Click **"Add a new pending publisher"**
3. Fill in:
   - **Owner:** `infinit3labs`
   - **Repository name:** `web-research-mcp`
   - **Workflow filename:** `publish.yml`
   - **Environment name:** `pypi` (must match the workflow's `environment:` value)
4. Click **Add**
5. Optional: repeat for the PyPI **Test** instance at https://test.pypi.org/manage/account/publishing/ if you want a staging check

That's it. No token to copy. PyPI now trusts any `publish` workflow run from this repo.

### 2. (Optional) Create the `pypi` environment in GitHub

This is the manual approval gate for production publishes.

1. Repo → Settings → Environments → **New environment**
2. Name: `pypi`
3. (Optional) Add yourself as a required reviewer so you have to click "Approve" before each release
4. Save

## Cutting a release

```bash
# 1. Make sure CHANGELOG.md is updated and pyproject.toml version is bumped
# 2. Commit the version bump
git add CHANGELOG.md pyproject.toml
git commit -m "chore(release): v0.2.0"
git push origin main

# 3. Create the release tag
gh release create v0.2.0 \
  --title "v0.2.0" \
  --notes "See CHANGELOG.md for details."

# GitHub Actions runs:
#   1. build job   → builds sdist + wheel
#   2. publish job → OIDC auth → uploads to PyPI
```

The package becomes installable within ~1 minute as:

```bash
pip install web-research-mcp
```

## Pre-release versions

For alpha/beta releases, follow [PEP 440](https://peps.python.org/pep-0440/):

```bash
# pyproject.toml
version = "0.2.0a1"  # or 0.2.0b1, 0.2.0rc1

gh release create v0.2.0a1 --prerelease --title "v0.2.0a1" --notes "..."
```

PyPI will mark it as a pre-release and `pip install web-research-mcp` will still install the latest stable. Users have to explicitly opt in with `pip install --pre`.

## Verifying a release

```bash
pip index versions web-research-mcp   # all versions
pip install --upgrade web-research-mcp # install latest
web-research-mcp --help               # sanity check (will hang waiting for stdio; Ctrl-C to exit is fine)
```

Or programmatically:

```bash
python -c "import web_research; print(web_research.__version__)"
```

## Rotating the publisher

If the GitHub repo is transferred to a new owner, remove the pending publisher from https://pypi.org/manage/account/publishing/ and re-register with the new owner.
