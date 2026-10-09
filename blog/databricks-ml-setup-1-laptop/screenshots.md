# Screenshot shot list (Part 1)

Two UI screenshots are missing from the post. Each one has a placeholder marked by a
`<!-- TODO(screenshot) -->` comment.

## General rules (same for Part 2)

- **Folder:** this post's `images/`, saved under the exact file name below (the post already
  links to it).
- **Format:** PNG, light theme, browser zoom at 100%, window roughly 1440 px wide. Crop to the
  part of the page that matters, without the browser chrome.
- **After saving**, set the figure's `{:data-width="..." data-height="..."}` in the post to the
  PNG's real pixel size. The placeholders say 1440 x 900.
- **Sanitise** (Appendix B of the skeleton). Blur or crop out:
  - the workspace URL / host (address bar, any links)
  - workspace, metastore and account IDs; run and experiment IDs in URLs or breadcrumbs
  - your email or user name, including the "Created by" / "Registered by" / "Owner" fields
- Use the **dev** objects (`taxi_dev`).

## Fig. 4: `fig04-mlflow-run.png` (MLflow run)

1. Sidebar: **Experiments** -> `/Shared/nyc_taxi_taxi_dev`.
2. Open the newest run named `taxi_fare_train`.
3. On the run's overview, capture **Parameters** and **Metrics** together if they fit. If they
   don't, filter the parameters table with `data_`, so it shows `data_table`,
   `data_start_batch_id`, `data_end_batch_id` and `data_table_version`.
4. The metrics must show `mae` (about 1.58) and `baseline_mae` (about 4.45).
5. Blur: "Created by", and any run or experiment IDs in the breadcrumb.

## Fig. 5: `fig05-model-version-champion.png` (model in UC with `@champion`)

1. Sidebar: **Catalog** -> `workspace` -> `taxi_dev` -> **Models** -> `taxi_fare_model`.
2. Capture the versions list with the **aliases** column showing `@champion` on the latest
   version (several versions in view is better: it shows that each retrain adds one).
3. Blur: owner / "Registered by", and the source run links if they expose IDs.
