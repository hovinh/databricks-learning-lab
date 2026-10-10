# Blog drafts

Drafts follow `references/blog-style.md`: one folder per post, `YYYY-MM-DD-<slug>.md` inside it,
figures in `images/`.

## Posts

| Folder | Status | Covers |
|---|---|---|
| `databricks-ml-setup/` | full draft (2026-10-10, replaces the two-part draft of 2026-10-09); 4 screenshots pending (`screenshots.md`) | Prototype to scheduled job in three stages: plain Python prototype, Databricks concepts + working config, layering rule + `Env`, MLflow + UC registry, hybrid via Connect, offline tests, Asset Bundles, SQL release runner, four environments + service principals |

## Candidate topics

Upgrade paths the databricks-ml-setup post names but doesn't cover:

- Wheel packaging with bundle `artifacts` + `python_wheel_task` (removes the `__file__` bootstrap).
- Champion/challenger promotion: only move `@champion` when the new version wins on the same test window.
- Monitoring with `vw_latest_predictions`: actual vs predicted over time, drift, alerting.
- CI/CD with GitHub Actions: `invoke all` + `bundle validate` on PR, `bundle deploy -t prd` on merge.
- Feature tables as Lakeflow declarative pipelines instead of a `.py` task.
