# Releasing

## Version numbers

Releases use CalVer in the form `YYYY.M.D.N`. `N` counts releases within a day
and starts at 1. Tags add a `v` prefix, so a second release on 1 September 2026
is tagged `v2026.9.1.2`.

Do not pad the month or the day with zeros. PEP 440 strips leading zeros, so a
tag of `v2026.09.01.2` reaches PyPI as `2026.9.1.2` and the tag stops matching
the version users see. Tags up to and including `v2026.08.21.2` predate this
rule and stay as they are.

## When to release

Release when a change reaches users. That means `src/hextraj/`, or packaging
metadata in `pyproject.toml` that PyPI displays.

`README.md` reaches users too, because `readme = "README.md"` makes it the PyPI
long description. A README-only change rides along with the next release rather
than earning one of its own.

Changes under `dev/`, `tests/`, and `.github/` never reach users, so they call
for no release at all. Let them accumulate on `main`.

## Cutting a release

A release is two publishing events. `.github/workflows/release.yml` covers only
the first one.

1. Bump `version` and `date-released` in `CITATION.cff`, and merge that through
   a pull request. `setuptools_scm` derives the package version from the git
   tag, but these two fields are hand-maintained.

2. Push the tag:

   ```shell
   git tag -a v2026.9.1.1 -m "Release v2026.9.1.1: <summary>"
   git push origin v2026.9.1.1
   ```

   The `push: tags: v*` trigger builds the distribution and publishes it to
   PyPI through a Trusted Publisher.

3. Create the GitHub release from that tag:

   ```shell
   gh release create v2026.9.1.1 --title v2026.9.1.1 --notes "<notes>"
   ```

   Zenodo hooks the release event, not the tag push. Skipping this step
   publishes to PyPI without minting a DOI. A draft release does not trigger
   Zenodo either.

## Checking the result

Read the Zenodo record back instead of assuming `.zenodo.json` was honoured:

```shell
curl -s "https://zenodo.org/api/records?q=hextraj&all_versions=true" \
    | python -m json.tool
```

Confirm `license`, `creators`, and `related_identifiers`. `.zenodo.json` takes
precedence over `CITATION.cff`, and it suppresses the license and the
repository link that Zenodo otherwise detects on its own.

## DOIs

Concept DOI `10.5281/zenodo.22040740` resolves to the most recent release.
Zenodo mints a separate DOI for each release. `README.md`, `CITATION.cff`, and
`[project.urls]` all record the concept DOI, so none of them needs an update
per release.
