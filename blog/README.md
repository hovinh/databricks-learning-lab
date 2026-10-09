# Blog drafts

Drafts follow `references/blog-style.md`: one folder per post, `YYYY-MM-DD-<slug>.md` inside it,
figures in `images/`.

## Posts

| Folder | Status | Covers |
|---|---|---|
| `databricks-ml-setup-1-laptop/` | full draft (2026-10-09); 2 screenshots pending (`screenshots.md`) | Part 1: versions, auth, pip-tools, layering rule, three stages, helpers, offline tests, MLflow + UC registry |
| `databricks-ml-setup-2-jobs/` | full draft (2026-10-09); 2 screenshots pending (`screenshots.md`) | Part 2: Asset Bundles, after-deploy gotchas, SQL release runner, rerunnable batch pipelines, multi-environment + service principals |

## Candidate topics

Upgrade paths the databricks-ml-setup series names but doesn't cover:

- Wheel packaging with bundle `artifacts` + `python_wheel_task` (removes the `__file__` bootstrap).
- Champion/challenger promotion: only move `@champion` when the new version wins on the same test window.
- Monitoring with `vw_latest_predictions`: actual vs predicted over time, drift, alerting.
- CI/CD with GitHub Actions: `invoke all` + `bundle validate` on PR, `bundle deploy -t prd` on merge.
- Feature tables as Lakeflow declarative pipelines instead of a `.py` task.
