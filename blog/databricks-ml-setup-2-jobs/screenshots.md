# Screenshot shot list (Part 2)

Two UI screenshots are missing from the post. Each one has a placeholder marked by a
`<!-- TODO(screenshot) -->` comment. The general rules (folder, format, sizing, what to blur)
are in Part 1's `screenshots.md`. On top of those, blur the user-name part of every
`[dev <user>]` job prefix, keeping the `[dev ` bracket visible.

## Fig. 3: `fig03-job-run-dag.png` (job run graph)

1. Sidebar: **Jobs & Pipelines** -> `[dev <user>] taxi_predict` -> **Runs**.
2. Open a successful run and switch to the **graph** view: `feature_engineering` -> `predict`,
   both green.
3. Optional: click `predict` so the side panel shows its parameters, including the resolved
   `--batch-id` (the task value).
4. Blur: the user name in the job name, and the run ID / URL.

## Fig. 4: `fig04-jobs-list-dev-prefix.png` (jobs list)

1. Sidebar: **Jobs & Pipelines**, filter by `taxi`.
2. Capture the list showing the three `[dev <user>] ...` jobs. If the prd jobs are deployed,
   keep them in the same shot: real names with no prefix and a **Paused** trigger, next to the
   dev ones. That makes the development-mode point by itself.
3. Blur the user-name part of each `[dev ...]` prefix.

## Optional

- The `log_sql_release` audit table in Catalog Explorer (Tables Are Not Code section). The post
  already shows it as a text capture, so this is only needed if you want one more UI image.
