# Publishing `torchleet`

Uses PyPI **trusted publishing**, so no API token is ever stored in the repo.

## One-time setup (repo owner)

1. Sign in at https://pypi.org and go to **Your projects → Publishing**
   (direct: https://pypi.org/manage/account/publishing/).
2. Under "Add a new pending publisher", fill in:
   - PyPI Project Name: `torchleet`
   - Owner: `Exorust`
   - Repository name: `TorchLeet`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In GitHub → Settings → Environments, create an environment named `pypi`.

## Releasing

```bash
# bump version in python/pyproject.toml first
git tag py-v0.1.0
git push origin py-v0.1.0
```

The `publish` workflow builds and uploads. Verify:

```bash
pip install torchleet
python -c "from torchleet import check, hint, status; status()"
```
